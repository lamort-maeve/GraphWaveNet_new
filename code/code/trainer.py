"""Server training entry point. --prepare-only never trains; --smoke is bounded."""
import argparse
import os
import csv
import json
import inspect
import hashlib
import random
import shutil
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler, SequentialSampler, BatchSampler

from data_pipeline import Samples, digest, load_prepared
from forecast_model import PaperGraphWaveNet
from metrics import Metrics, masked_mae
from graph_controls import graph_control
from matrix_io import load_matrix, reference_supports
from mean_teacher import (unwrap, is_constant_buffer, make_teacher, update_teacher,
                          perturb_inputs, consistency_loss)
import pi_model
from temporal_ensembling import TemporalEnsembling
from self_training import SelfTraining
from execution import BatchSamples

ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'data/data_full_year')  # train/val/test位置
    p.add_argument('--raw', type=Path, default=ROOT/'data/full_year_cleaned.npz')  # 原始NPZ位置
    p.add_argument('--graphs', type=Path, default=ROOT/'per_window')  # PCMCI图目录
    p.add_argument('--prepared-pilot', type=Path, help='Audited prepared pilot root; resolves raw/split/graph paths and verifies provenance')
    p.add_argument('--graph-availability', choices=['online', 'frozen_train'], default='frozen_train',
                   help='Online as-of graphs or freeze discovery at the last retained training origin')
    p.add_argument('--output', type=Path, default=ROOT/'results/paper')  # 结果保存位置
    p.add_argument('--mode', choices=['causal', 'adaptive', 'combined', 'custom', 'all'], default='all')
    p.add_argument('--variables', choices=['all', 'flow'], default='all')
    p.add_argument('--device', default='auto', help='auto selects CUDA when available; explicit cuda fails if unavailable')
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--architecture', choices=['paper'], default='paper')
    p.add_argument('--graph-control', choices=['dynamic','static','hybrid','shuffled','collapsed','local','zero','unsigned'], default='dynamic',
                   help='static: first lagged graph; hybrid: static+dynamic (external prior optional); local: self edges')
    p.add_argument('--label-ratio', type=float, default=1., help='Fraction of training forecast origins receiving supervision; historical observations remain available')
    p.add_argument('--semi-supervised', action='store_true', help='EMA teacher consistency on unsupervised training origins (not a missing-sensor protocol)')
    p.add_argument('--ssl-method', choices=['none', 'mean_teacher', 'pi_model', 'temporal_ensembling', 'self_training'], default='none',
                   help='Explicit SSL baseline; --semi-supervised remains a compatibility alias')
    p.add_argument('--static-matrix', type=Path, help='External spatial prior for hybrid; never used by adaptive/custom')
    p.add_argument('--static-available-index', type=int, help='First raw timeline index at which the external prior was available')
    p.add_argument('--custom-matrix', type=Path, help='Original GraphWaveNet support matrix; requires --mode custom')
    p.add_argument('--custom-adaptive', action='store_true', help='Original fixed supports plus original learned adaptive adjacency')
    p.add_argument('--adjtype', choices=['raw', 'transition', 'doubletransition', 'identity'], default='doubletransition')
    p.add_argument('--matrix-key', default='adjacency', help='NPZ/dict matrix array key')
    p.add_argument('--matrix-assume-order', action='store_true', help='Explicitly assert dataset order for matrix files without node IDs')
    p.add_argument('--trusted-matrix-pickle', action='store_true')
    p.add_argument('--graph-kernel', type=int, default=1, help='1: original dynamic reader; >1: v2-inspired past graph fusion')
    p.add_argument('--history-fusion', choices=['learned', 'uniform', 'contextual', 'shrunk'], default='learned')
    p.add_argument('--lag-readout', choices=['aligned', 'synchronous'], default='aligned',
                   help='synchronous removes only the lagged feature shift; graph lag channels remain')
    p.add_argument('--history-weight-sharing', choices=['per_lag', 'shared'], default='per_lag')
    p.add_argument('--dynamic-message-scale', type=float, default=1.,
                   help='Scale dynamic messages before static fusion; zero isolates the static branch')
    p.add_argument('--causal-revision', choices=['legacy', 'selective'], default='legacy')
    p.add_argument('--adaptive-scale', type=float, default=1., help='Combined-only adaptive edge scale; zero-hop local transform remains')
    p.add_argument('--history-uniform-mix', type=float, default=.5)
    p.add_argument('--ssl-clean-labeled', action='store_true', help='Keep labeled student inputs clean; apply identically to MT controls')
    p.add_argument('--history-init', choices=['recent', 'uniform'], default='recent')
    p.add_argument('--history-padding', choices=['zero', 'masked'], default='zero')
    p.add_argument('--lag-normalization', choices=['all', 'valid'], default='all')
    p.add_argument('--reliability-history', type=int, default=3)
    p.add_argument('--reliability-gamma', type=float, default=2.)
    p.add_argument('--diagnose-teacher', action='store_true', help='Evaluate withheld training labels after fitting; exploratory only')
    p.add_argument('--diagnostic-fraction', type=float, default=0., help='Reserve a purged chronological tail inside train before fitting the scaler')
    p.add_argument('--train-start-index', type=int, default=0, help='Common minimum training origin for prefix-prior comparisons')
    p.add_argument('--static-provenance', type=Path, help='Required JSON provenance for an external static prior')
    p.add_argument('--static-control', choices=['real', 'random', 'zero'], default='real')
    p.add_argument('--causal-execution', choices=['reference', 'signed_cache', 'persistent_cache'], default='reference',
                   help='signed_cache shares fixed sign splitting across layers; validate throughput on your GPU before enabling')
    p.add_argument('--static-fusion', choices=['convex', 'gated_residual'], default='convex')
    p.add_argument('--static-weight', type=float, default=0.5,
                   help='Hybrid only: static message weight; external spatial matrix if provided, otherwise first lagged graph')
    p.add_argument('--ssl-weight', type=float, default=0.1)
    p.add_argument('--ssl-warmup', type=int, default=10)
    p.add_argument('--ema-decay', type=float, default=0.99)
    p.add_argument('--input-noise', type=float, default=0.02, help='Student input noise in normalized units; teacher sees clean histories')
    p.add_argument('--pi-weight', type=float, default=0.1, help='Pi-model normalized MSE weight; independent of Mean Teacher ssl-weight')
    p.add_argument('--pi-noise', type=float, default=0.02)
    p.add_argument('--pi-warmup', type=int, default=10)
    p.add_argument('--te-weight', type=float, default=0.1)
    p.add_argument('--te-noise', type=float, default=0.02)
    p.add_argument('--te-warmup', type=int, default=10)
    p.add_argument('--te-decay', type=float, default=0.6)
    p.add_argument('--st-weight', type=float, default=0.1)
    p.add_argument('--st-noise', type=float, default=0.02)
    p.add_argument('--st-pretrain', type=int, default=10, help='Supervised-only epochs before pseudo-label training')
    p.add_argument('--st-warmup', type=int, default=10)
    p.add_argument('--st-refresh', type=int, default=5, help='Hard snapshot refresh interval, not EMA')
    p.add_argument('--graph-weighting', choices=['confidence', 'legacy'], default='confidence',
                   help='PCMCI edge weighting; keep identical across matched comparisons')
    p.add_argument('--trusted-object-metadata', action='store_true', help='Allow object dtype metadata in your own trusted NPZ')
    p.add_argument('--resume', action='store_true', help='Resume only a matching current run')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--blocks', type=int, default=4)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--context', type=int, default=25)
    p.add_argument('--lr-patience', type=int, default=5)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--threads', type=int, default=0, help='0: up to 8 CPU threads within process affinity; positive: explicit count')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--batch-fetch', choices=['sample', 'vectorized'], default='sample', help='Vectorized CPU batch fetch, preserving sampler order')
    p.add_argument('--cache-dir', type=Path, default=None, help='Prepared input cache; default OUTPUT/cache')
    p.add_argument('--no-data-cache', action='store_true', help='Read and validate original inputs on every launch')
    p.add_argument('--rebuild-cache', action='store_true', help='Force revalidation and replacement of prepared input cache')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--skip-test', action='store_true', help='Validation-only tuning: save best checkpoint without evaluating test data')
    p.add_argument('--final-test', action='store_true', help='Explicitly authorize test evaluation for a frozen prepared-pilot configuration')
    p.add_argument('--smoke', action='store_true', help='Only 2 train batches and 2 eval batches per mode')
    p.add_argument('--run-name', default=None)
    return p


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolved_device(value):
    return ('cuda:0' if torch.cuda.is_available() else 'cpu') if value == 'auto' else value


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def load_checkpoint(path, device):
    kwargs = {'map_location': device}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = True
    return torch.load(path, **kwargs)


def atomic_save(value, path):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def atomic_copy(source, destination):
    """Reuse the already serialized checkpoint; no second CUDA tensor download."""
    temp = destination.with_suffix('.tmp')
    shutil.copyfile(source, temp)
    temp.replace(destination)


def checkpoint_model_state(model, constant_buffers):
    state = unwrap(model).state_dict()
    # These buffers never change during optimization. In particular graph_bank
    # can be much larger than all learned weights and need not leave CUDA again.
    for name, value in constant_buffers.items():
        state[name] = value
    return state


def pass_epoch(model, loader, device, mean, std, optimizer=None, capture=False,
               teacher=None, ssl_weight=0., input_noise=0., ema_decay=0.99,
               ssl_method='none', ssl_state=None, ssl_clean_labeled=False):
    if teacher is not None and ssl_method not in ('none', 'mean_teacher'):
        raise ValueError('Independent SSL methods must not use a Mean Teacher instance')
    if ssl_method in ('temporal_ensembling', 'self_training') and ssl_state is None:
        raise ValueError('This SSL method requires its own training state')
    if ssl_state is not None and (teacher is not None or ssl_method not in ('temporal_ensembling', 'self_training')):
        raise ValueError('Independent SSL state cannot be combined with another strategy')
    training = optimizer is not None
    model.train(training)
    metric = Metrics(getattr(loader, 'forecast_horizon', None) or len(loader.dataset[0][1]))
    predictions, targets, masks, origins = [], [], [], []
    finite = torch.ones((), dtype=torch.bool, device=device)
    parameters = list(model.parameters())
    loss_totals = torch.zeros(3, device=device)
    with torch.set_grad_enabled(training):
        for x, y, valid, graph_ids, origin in loader:
            labeled = loader.supervision[origin].view(-1, 1, 1) if training and hasattr(loader, 'supervision') else torch.ones((len(origin), 1, 1), dtype=torch.bool)
            if training:
                valid = valid & labeled
            if training and not bool(valid.any()) and teacher is None and ssl_method not in ('pi_model', 'temporal_ensembling') and not (ssl_method == 'self_training' and ssl_state.active):
                continue
            cpu_y, cpu_valid = y, valid
            x, y, valid, graph_ids = (v.to(device, non_blocking=True) for v in (x, y, valid, graph_ids))
            clean_x = x
            if training and teacher is not None:
                x = perturb_inputs(x, input_noise)
                if ssl_clean_labeled:
                    x = torch.where(labeled.to(device).view(-1, 1, 1, 1), clean_x, x)
            elif training and ssl_method == 'pi_model':
                x = pi_model.perturb(x, input_noise)
            elif training and ssl_state is not None and ssl_state.active:
                x = ssl_state.perturb(x, input_noise)
            output = model(x, graph_ids)
            pred = output * std + mean
            finite.logical_and_(torch.isfinite(pred).all())
            if training:
                supervised = masked_mae(pred, y, valid)
                consistency = pred.new_zeros(())
                if teacher is not None:
                    consistency = consistency_loss(teacher, clean_x, graph_ids, pred, labeled, mean, std)
                elif ssl_method == 'pi_model':
                    second = model(pi_model.perturb(clean_x, input_noise), graph_ids)
                    consistency = pi_model.consistency_loss(output, second, labeled)
                elif ssl_state is not None:
                    consistency = ssl_state.loss(output, clean_x, graph_ids, origin, labeled)
                loss = supervised + ssl_weight*consistency
                finite.logical_and_(torch.isfinite(loss))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Do not convert the CUDA norm to a Python bool every batch.
                norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=False)
                finite.logical_and_(torch.isfinite(norm))
                optimizer.step()
                if teacher is not None:
                    update_teacher(teacher, model, ema_decay)
                loss_totals += torch.stack([supervised.detach(), consistency.detach(), pred.new_ones(())])
            metric.update(pred, y, valid)
            if capture:
                predictions.append(pred.detach().cpu().numpy())
                targets.append(cpu_y.numpy())
                masks.append(cpu_valid.numpy())
                origins.append(origin.numpy())
    # Check before reporting metrics or saving any checkpoint. Failure detection
    # is deferred to the end of this pass, rather than synchronizing each batch.
    if not bool(finite):
        raise FloatingPointError('Nonfinite predictions/gradient norm; pass aborted without saving a checkpoint')
    packed = None
    if capture:
        packed = dict(prediction=np.concatenate(predictions), target=np.concatenate(targets),
                      observed=np.concatenate(masks), origin_index=np.concatenate(origins))
    result = metric.result()
    if training:
        supervised, consistency, batches = loss_totals.cpu().tolist()
        result['objective'] = dict(supervised_batch_mae=supervised/max(1, batches),
                                   consistency_batch_mae=consistency/max(1, batches), ssl_weight=ssl_weight,
                                   optimizer_steps=int(batches))
        if ssl_method in ('pi_model', 'temporal_ensembling', 'self_training'):
            result['objective']['consistency_normalized_mse'] = result['objective'].pop('consistency_batch_mae')
    return result, packed


def make_loader(data, archive, split, args):
    dataset = Samples(data, archive, split)
    if args.smoke:
        # Sample across the timeline, including late graphs, not only the graph-free warmup.
        indices = np.linspace(0, len(dataset)-1, min(len(dataset), args.batch_size*2), dtype=int)
        dataset = Subset(dataset, indices.tolist())
    generator = torch.Generator().manual_seed(args.seed)
    # Separate worker seeding from sampling: persistent workers should not change
    # the next epoch's sample order or consume the sampler's checkpointed RNG.
    sampler = RandomSampler(dataset, generator=generator) if args.workers and split == 'train' else None
    if getattr(args, 'batch_fetch', 'sample') == 'vectorized':
        sample_sampler = sampler or (RandomSampler(dataset, generator=generator) if split == 'train' else SequentialSampler(dataset))
        loader = DataLoader(BatchSamples(dataset), batch_size=None,
            sampler=BatchSampler(sample_sampler, args.batch_size, drop_last=False),
            num_workers=args.workers, persistent_workers=args.workers > 0,
            pin_memory=torch.device(resolved_device(args.device)).type == 'cuda',
            generator=torch.Generator().manual_seed(args.seed + 1) if args.workers else generator)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=split == 'train' and sampler is None,
                          sampler=sampler,
                          num_workers=args.workers, drop_last=False,
                          persistent_workers=args.workers > 0,
                          pin_memory=torch.device(resolved_device(args.device)).type == 'cuda',
                          generator=torch.Generator().manual_seed(args.seed + 1) if args.workers else generator)
    loader.forecast_horizon = data.horizon if hasattr(data, 'horizon') else data.targets[split].shape[1]
    if split == 'train':
        # Fixed, exact sample-level budget, independent of shuffle order/epochs.
        origins = data.splits[split]
        candidates = origins[np.asarray(data.masks[split]).reshape(len(origins), -1).any(1)]
        if not len(candidates):
            raise ValueError('No observed training labels')
        count = max(1, int(np.ceil(len(candidates)*getattr(args, 'label_ratio', 1.))))
        selected = np.random.default_rng(args.seed).choice(candidates, count, replace=False)
        loader.supervision = torch.zeros(int(origins.max())+1, dtype=torch.bool)
        loader.supervision[selected] = True
        loader.label_origins = sorted(map(int, selected))
    return loader


def loader_rng(loader):
    sampler = loader.sampler.sampler if isinstance(loader.sampler, BatchSampler) else loader.sampler
    return sampler.generator if isinstance(sampler, RandomSampler) else loader.generator


def run_mode(args, data, archive, directory, mode, diagnostic_archive=None):
    seed_all(args.seed)
    args.device = resolved_device(args.device)
    run = directory/mode
    run.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device(args.device)
    matrix_options = dict(trusted_pickle=args.trusted_matrix_pickle,
                          assume_order=args.matrix_assume_order, key=args.matrix_key)
    static_matrix, supports = None, None
    minimum_graph = 1
    if mode == 'custom':
        matrix = load_matrix(args.custom_matrix, data.node_ids, data.features, args.variables, **matrix_options)
        supports = reference_supports(matrix, args.adjtype)
    elif mode != 'adaptive' and args.static_matrix:
        matrix = load_matrix(args.static_matrix, data.node_ids, data.features, args.variables, **matrix_options)
        if args.static_control == 'random':
            # Same edge weights/count and projection capacity, random locations.
            matrix = np.random.default_rng(args.seed).permutation(matrix.reshape(-1)).reshape(matrix.shape)
        elif args.static_control == 'zero':
            # Retain the static projection and identical initialization/RNG,
            # but remove static information. Provenance checks still apply.
            matrix = np.zeros_like(matrix)
        static_matrix = matrix / np.maximum(np.abs(matrix).sum(0, keepdims=True), 1.)
        # Conservative availability: activate at the first dynamic snapshot at
        # or after the declared static completion time, never earlier.
        minimum_graph = int(np.searchsorted(archive.times, args.static_available_index, side='left')) + 1
        if minimum_graph > len(archive.times):
            raise ValueError('Static prior is unavailable throughout the graph archive')
    model = PaperGraphWaveNet(
        data.values.shape[1], data.horizon, data.flow_indices, archive.bank, mode,
        args.hidden, args.blocks, args.layers, args.dropout,
        static_weight=args.static_weight if args.graph_control == 'hybrid' and mode not in ('adaptive', 'custom') else 0.,
        static_matrix=static_matrix, static_min_graph_id=minimum_graph, graph_kernel=args.graph_kernel,
        custom_supports=supports, custom_adaptive=args.custom_adaptive,
        causal_execution=args.causal_execution, history_fusion=args.history_fusion,
        history_padding=args.history_padding, lag_normalization=args.lag_normalization, history_init=args.history_init,
        causal_revision=args.causal_revision, adaptive_scale=args.adaptive_scale, uniform_mix=args.history_uniform_mix, static_fusion=args.static_fusion,
        lag_readout=args.lag_readout, history_weight_sharing=args.history_weight_sharing,
        dynamic_message_scale=args.dynamic_message_scale
    ).to(device)
    if args.gpus != 1:
        raise ValueError('One GPU per experiment preserves the BatchNorm batch semantics')
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                          patience=args.lr_patience, min_lr=1e-5)
    splits = ('train', 'val') if args.skip_test else ('train', 'val', 'test')
    loaders = {s: make_loader(data, archive, s, args) for s in splits}
    np.save(run/'labeled_origins.npy', np.asarray(loaders['train'].label_origins, dtype=np.int64))
    teacher = None
    if args.ssl_method == 'mean_teacher' or (args.semi_supervised and args.ssl_method == 'none'):
        teacher = make_teacher(model)
    ssl_state = None
    if args.ssl_method == 'temporal_ensembling':
        ssl_state = TemporalEnsembling(len(loaders['train'].supervision), args.te_decay, device)
    elif args.ssl_method == 'self_training':
        ssl_state = SelfTraining(args.st_pretrain, args.st_refresh)
    mean = torch.as_tensor(data.output_mean, device=device)
    std = torch.as_tensor(data.output_std, device=device)
    config = dict(vars(args), mode=mode, context=data.context, horizon=data.horizon,
                  ssl_algorithm=(args.ssl_method+'_regression' if teacher is not None else
                                 args.ssl_method+'_regression' if args.ssl_method != 'none' else 'supervised'),
                  evaluation_model='student',
                  implementation='paper_static_lag_v5_20261026',
                  supervision_protocol='forecast_origin_subsampling_with_observed_histories',
                  supervised_origins=len(loaders['train'].label_origins),
                  supervision_mask_sha256=hashlib.sha256(loaders['train'].supervision.numpy().tobytes()).hexdigest(),
                  target_offsets=data.y_offsets.tolist(), frozen_adaptive_backbone=False,
                  code_hashes={name: digest(Path(__file__).parent/name) for name in
                               ('trainer.py', 'data_pipeline.py', 'forecast_model.py', 'graphwavenet.py',
                                'mean_teacher.py', 'pi_model.py', 'temporal_ensembling.py', 'self_training.py', 'metrics.py', 'graph_controls.py', 'matrix_io.py', 'causal_fusion.py', 'causal_revision.py', 'execution.py', 'graph_reliability.py', 'static_provenance.py', 'teacher_diagnostics.py', 'pilot_protocol.py', 'graph_protocol.py')},
                  variables_count=data.values.shape[1], parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  layer_history_lengths=[max(data.context, model.receptive_field)-sum(
                      2**(j % args.layers) for j in range(i)) for i in range(args.blocks*args.layers)],
                  receptive_field=int(unwrap(model).receptive_field),
                  temporal_kernel=int(getattr(unwrap(model), 'temporal_kernel', 2)),
                  python_version=platform.python_version(), cuda_version=torch.version.cuda,
                  device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                  cudnn_version=torch.backends.cudnn.version(),
                  torch_version=str(torch.__version__), numpy_version=np.__version__)
    config = json.loads(json.dumps(config, default=str))
    best, stale, history = float('inf'), 0, []
    start_epoch = 1
    if args.resume and (run/'last.pt').exists():
        saved = load_checkpoint(run/'last.pt','cpu')
        for key in ('code_hashes', 'input_provenance', 'adaptive_checkpoint_sha256', 'torch_version', 'numpy_version'):
            if key in saved['config'] and saved['config'][key] != config.get(key):
                raise ValueError(f'Resume provenance mismatch: {key}; start a new run')
        for key in ('architecture','variables','context','hidden','blocks','layers','dropout',
                    'graph_weighting','seed','alpha','batch_size','learning_rate','weight_decay','lr_patience',
                    'graph_control','static_weight','static_available_index','graph_kernel','causal_execution','adjtype','custom_adaptive','label_ratio','semi_supervised','ssl_method',
                    'causal_revision','adaptive_scale','history_uniform_mix','ssl_clean_labeled',
                    'batch_fetch','static_fusion',
                    'lag_readout','history_weight_sharing','dynamic_message_scale',
                    'history_fusion','history_init','history_padding','lag_normalization','reliability_history','reliability_gamma',
                    'train_start_index','static_control','diagnostic_fraction','graph_availability',
                    'ssl_weight','ssl_warmup','ema_decay','input_noise','pi_weight','pi_noise','pi_warmup',
                    'te_weight','te_noise','te_warmup','te_decay',
                    'st_weight','st_noise','st_pretrain','st_warmup','st_refresh'):
            if saved['config'].get(key) != config.get(key):
                raise ValueError(f'Resume configuration mismatch: {key}')
        for key, expected in [('scaler_mean', data.mean), ('scaler_std',data.std), ('graph_times',archive.times)]:
            if not np.array_equal(saved[key].cpu().numpy(),expected):
                raise ValueError(f'Resume data mismatch: {key}')
        unwrap(model).load_state_dict(saved['model'])
        if teacher is not None:
            teacher.load_state_dict(saved['teacher'])
        if ssl_state is not None:
            ssl_state.load_state_dict(saved['ssl_state'], model)
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        best,stale,history=saved['best'],saved['stale'],saved['history']
        if stale == 0:
            # Recover an interruption between committing last.pt and copying best.
            atomic_copy(run/'last.pt', run/'best.pt')
        start_epoch=saved['epoch']+1
        torch.set_rng_state(saved['torch_rng'].cpu())
        if torch.cuda.is_available() and saved['cuda_rng']:
            torch.cuda.set_rng_state_all([t.cpu() for t in saved['cuda_rng']])
        loader_rng(loaders['train']).set_state(saved['loader_rng'].cpu())
    write_json(run/'config.json', config)
    # Prepare fixed checkpoint metadata once, including after checkpoint restore.
    constant_buffers = {name: value.detach().cpu().clone() for name, value in unwrap(model).named_buffers()
                        if is_constant_buffer(name)}
    scaler_mean, scaler_std = torch.from_numpy(data.mean), torch.from_numpy(data.std)
    graph_times = torch.from_numpy(archive.times)
    history_path = run/'history.csv'
    if history:
        # Resume may follow an interrupted epoch whose CSV row was written before
        # last.pt. Reconcile once from the committed checkpoint, then append.
        with open(history_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader(); writer.writerows(history)
    for epoch in range(start_epoch, (1 if args.smoke else args.epochs)+1):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.time()
        weight = args.ssl_weight*min(1., epoch/max(1, args.ssl_warmup)) if teacher is not None else 0.
        noise = args.input_noise
        if args.ssl_method == 'pi_model':
            weight = args.pi_weight*min(1., epoch/max(1, args.pi_warmup))
            noise = args.pi_noise
        if ssl_state is not None:
            ssl_state.start_epoch(epoch, model)
            if args.ssl_method == 'temporal_ensembling':
                weight = args.te_weight*min(1., (epoch-1)/max(1, args.te_warmup))
                noise = args.te_noise
            else:
                weight = args.st_weight*min(1., max(0, epoch-args.st_pretrain)/max(1, args.st_warmup))
                noise = args.st_noise
        train, _ = pass_epoch(model, loaders['train'], device, mean, std, optimizer,
                              teacher=teacher, ssl_weight=weight, input_noise=noise, ema_decay=args.ema_decay,
                              ssl_method=args.ssl_method, ssl_state=ssl_state,
                              ssl_clean_labeled=args.ssl_clean_labeled)
        if ssl_state is not None:
            ssl_state.end_epoch()
        val, _ = pass_epoch(model, loaders['val'], device, mean, std)
        score = val['overall']['mae']
        if score is None or not np.isfinite(score):
            raise ValueError('Validation has no finite observed-target metric')
        row = dict(epoch=epoch, seconds=time.time()-start, learning_rate=optimizer.param_groups[0]['lr'],
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                   **train['objective'],
                   **{'train_'+k: v for k, v in train['overall'].items()},
                   **{'val_'+k: v for k, v in val['overall'].items()})
        history.append(row)
        with open(history_path, 'a' if len(history) > 1 else 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if len(history) == 1:
                writer.writeheader()
            writer.writerow(row)
        improved = score < best
        if improved:
            best, stale = score, 0
        else:
            stale += 1
        scheduler.step(score)
        checkpoint = dict(model=checkpoint_model_state(model, constant_buffers), optimizer=optimizer.state_dict(),
                          scheduler=scheduler.state_dict(), epoch=epoch, validation=val, config=config,
                          best=best, stale=stale, history=history,
                          torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                          loader_rng=loader_rng(loaders['train']).get_state(),
                          scaler_mean=scaler_mean, scaler_std=scaler_std,
                          graph_times=graph_times)
        if teacher is not None:
            checkpoint['teacher'] = checkpoint_model_state(teacher, constant_buffers)
        if ssl_state is not None:
            checkpoint['ssl_state'] = ssl_state.state_dict()
        save_start = time.perf_counter()
        atomic_save(checkpoint, run/'last.pt')
        if improved:
            atomic_copy(run/'last.pt', run/'best.pt')
        print(f'{mode} epoch={epoch} train_MAE={train["overall"]["mae"]} '
              f'val_MAE={score:.4f} train+val_seconds={row["seconds"]:.1f} '
              f'save_seconds={time.perf_counter()-save_start:.1f}', flush=True)
        if args.patience and stale >= args.patience:
            break
    # Evaluation needs model weights only; do not upload the saved Adam state
    # and a second graph bank to CUDA just to immediately discard them.
    checkpoint = load_checkpoint(run/'best.pt', 'cpu')
    if args.diagnose_teacher:
        from teacher_diagnostics import diagnose
        teacher.load_state_dict(checkpoint['teacher'])
        diagnose(teacher, data, archive, loaders['train'].supervision, args, run, mean, std,
                 diagnostic_archive=diagnostic_archive)
    if args.skip_test:
        result = dict(mode=mode, best_epoch=checkpoint['epoch'],
                      best_validation_mae=checkpoint['validation']['overall']['mae'],
                      smoke_only=args.smoke, test_evaluated=False)
        write_json(run/'validation_selection.json', result)
        return result
    unwrap(model).load_state_dict(checkpoint['model'])
    test, arrays = pass_epoch(model, loaders['test'], device, mean, std, capture=True)
    test['best_epoch'] = checkpoint['epoch']
    test['best_validation_mae'] = checkpoint['validation']['overall']['mae']
    test['smoke_only'] = args.smoke
    test['pcmci_validation_improved'] = None
    test['target_offsets'] = data.y_offsets.tolist()
    write_json(run/'test_metrics.json', test)
    np.savez_compressed(run/'test_predictions.npz', **arrays)
    return dict(mode=mode, best_epoch=checkpoint['epoch'],
                best_validation_mae=test['best_validation_mae'], **test['overall'], smoke_only=args.smoke)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.prepared_pilot:
        from pilot_protocol import validate_pilot
        meta = validate_pilot(args.prepared_pilot, require_graphs=args.mode not in ('adaptive', 'custom'),
                              expected_protocol=args.graph_availability)
        args.raw = args.prepared_pilot/'cleaned.npz'
        args.data = args.prepared_pilot/'forecast'
        args.graphs = args.prepared_pilot/'graphs/per_window'
        if args.context != meta['context']:
            raise ValueError('Pilot context must match prepared context for identical split boundaries')
        if args.mode not in ('adaptive', 'custom'):
            graph_config = json.loads((args.prepared_pilot/'graphs/graph_config.json').read_text(encoding='utf-8'))
            if args.alpha != graph_config['alpha']:
                raise ValueError('Pilot alpha must match graph discovery configuration')
        if args.train_start_index != meta['static_available_index']:
            raise ValueError('Prepared pilot comparisons require the common manifest static_available_index as train-start-index')
        if args.final_test and args.skip_test:
            raise ValueError('Choose validation-only or final-test, not both')
        args.skip_test = not args.final_test
    if args.semi_supervised and args.ssl_method not in ('none', 'mean_teacher'):
        raise ValueError('--semi-supervised is a Mean Teacher alias; do not combine it with another SSL method')
    if args.semi_supervised:
        args.ssl_method = 'mean_teacher'
    args.semi_supervised = args.ssl_method == 'mean_teacher'
    if not all(0 <= v <= 1 for v in (args.adaptive_scale, args.history_uniform_mix)):
        raise ValueError('Invalid adaptive/history parameters')
    if args.adaptive_scale != 1 and args.mode != 'combined':
        raise ValueError('Reduced adaptive scale requires combined mode; baseline is unchanged')
    if args.causal_revision != 'legacy' and args.mode not in ('combined', 'causal'):
        raise ValueError('Selective injection requires a causal branch')
    if not 0 <= args.dynamic_message_scale <= 1:
        raise ValueError('dynamic-message-scale must be in [0,1]')
    if (args.lag_readout != 'aligned' or args.history_weight_sharing != 'per_lag' or args.dynamic_message_scale != 1) and args.mode not in ('combined', 'causal'):
        raise ValueError('Static/lag ablations require causal/combined mode')
    if args.history_fusion == 'contextual' and (args.lag_readout != 'aligned' or args.history_weight_sharing != 'per_lag'):
        raise ValueError('Static/lag readout controls are not defined for contextual fusion')
    if args.history_weight_sharing == 'shared' and (args.graph_kernel < 2 or args.history_fusion not in ('learned', 'shrunk')):
        raise ValueError('Shared weights require K>=2 and learned/shrunk fusion')
    if args.history_fusion in ('contextual', 'shrunk') and (args.mode not in ('combined', 'causal') or args.graph_kernel < 2 or args.history_padding != 'masked' or args.lag_normalization != 'valid' or args.history_init != 'uniform'):
        raise ValueError('Contextual fusion requires causal/combined, K>=2, masked/valid/uniform policies')
    if args.reliability_history < 1 or not np.isfinite(args.reliability_gamma) or args.reliability_gamma < 0:
        raise ValueError('Invalid graph reliability parameters')
    if args.diagnose_teacher and args.ssl_method != 'mean_teacher':
        raise ValueError('Teacher diagnostics require a teacher strategy')
    if args.train_start_index < 0:
        raise ValueError('train-start-index must be nonnegative')
    if not 0 <= args.diagnostic_fraction < .5:
        raise ValueError('diagnostic-fraction must be in [0, .5)')
    if min(args.pi_weight, args.pi_noise, args.pi_warmup) < 0:
        raise ValueError('Pi-model weight, noise and warmup must be nonnegative')
    if not 0 <= args.te_decay < 1 or min(args.te_weight, args.te_noise, args.te_warmup,
            args.st_weight, args.st_noise, args.st_warmup) < 0 or min(args.st_pretrain, args.st_refresh) < 1:
        raise ValueError('Invalid temporal ensembling or self-training configuration')
    if not 0 <= args.static_weight <= 1:
        raise ValueError('--static-weight must be in [0, 1]')
    if min(args.epochs, args.batch_size, args.hidden, args.blocks, args.layers, args.context, args.graph_kernel, args.gpus) < 1:
        raise ValueError('Training dimensions and epochs must be positive')
    if min(args.patience,args.lr_patience,args.workers,args.threads) < 0 or not 0 <= args.dropout < 1:
        raise ValueError('Invalid patience/workers/dropout')
    if not 0 < args.label_ratio <= 1 or not 0 <= args.ema_decay < 1 or min(args.ssl_weight, args.ssl_warmup, args.input_noise) < 0:
        raise ValueError('Invalid label budget or SSL configuration')
    if args.ssl_method != 'none' and args.label_ratio == 1:
        raise ValueError('SSL requires --label-ratio < 1; full-label runs use supervised learning')
    if args.mode in ('adaptive', 'custom') and args.graph_control != 'dynamic':
        raise ValueError('Adaptive baseline does not use graphs; graph controls are inapplicable')
    if (args.mode == 'custom') != (args.custom_matrix is not None):
        raise ValueError('--mode custom and --custom-matrix must be used together')
    if args.custom_adaptive and args.mode != 'custom':
        raise ValueError('--custom-adaptive only applies to custom mode')
    if args.static_matrix and (args.graph_control != 'hybrid' or args.mode in ('adaptive', 'custom')):
        raise ValueError('--static-matrix requires causal/combined/all mode with --graph-control hybrid')
    if args.static_matrix and (args.static_available_index is None or args.static_available_index < 0):
        raise ValueError('Declare --static-available-index >= 0 for the external prior')
    if args.static_fusion == 'gated_residual' and (not args.static_matrix or not args.static_weight):
        raise ValueError('Gated static residual needs an external prior and positive static weight')
    if args.static_matrix:
        from static_provenance import validate_static_provenance
        validate_static_provenance(args)
    elif args.static_provenance or args.static_control != 'real':
        raise ValueError('Static provenance/control requires --static-matrix')
    if args.static_available_index is not None and not args.static_matrix:
        raise ValueError('--static-available-index requires --static-matrix')
    if args.gpus != 1:
        raise ValueError('Use one GPU per experiment: DataParallel changes BatchNorm statistics and training results')
    if args.device == 'auto':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; install a CUDA-enabled PyTorch and check the driver')
    if args.threads == 0:
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        args.threads = min(8, cpus)
    torch.set_num_threads(args.threads)
    print(f'Runtime: device={device}; CPU threads={torch.get_num_threads()}; workers={args.workers}; '
          f'torch={torch.__version__}; CUDA build={torch.version.cuda}; '
          f'GPU={torch.cuda.get_device_name(device) if device.type == "cuda" else "none"}', flush=True)
    # Cache includes full alignment validation, graph preparation and input hashes.
    data, archive, inputs = load_prepared(args)
    if args.static_matrix and args.static_available_index > int(data.splits['train'][-1]):
        raise ValueError('Static prior becomes available after the retained training interval')
    # A training-only probe may measure structural drift on held-out histories.
    # These graphs are not supplied to the frozen teacher or any optimizer.
    diagnostic_archive = archive
    if args.train_start_index:
        keep = data.splits['train'] >= args.train_start_index
        if not keep.any():
            raise ValueError('No training samples remain after train-start-index')
        for mapping in (data.splits, data.targets, data.masks):
            mapping['train'] = mapping['train'][keep]
    if args.graph_availability == 'frozen_train' and len(archive.times):
        import copy
        archive = copy.copy(archive)
        count = int(np.searchsorted(archive.times, data.splits['train'][-1], side='right'))
        archive.times = archive.times[:count].copy()
        archive.bank = archive.bank[:count+1].copy()
        archive.raw = archive.raw[:count].copy()
        archive.metadata = dict(archive.metadata, windows=archive.metadata['windows'][:count], graph_count=count)
    archive = graph_control(archive, args.graph_control, args.seed)
    for kind, path in [('static_matrix', args.static_matrix), ('custom_matrix', args.custom_matrix),
                       ('static_provenance', args.static_provenance)]:
        if path:
            inputs[kind] = dict(path=str(path.resolve()), sha256=digest(path))
    if args.prepared_pilot:
        inputs['dataset_manifest'] = dict(path=str((args.prepared_pilot/'dataset_manifest.json').resolve()),
                                          sha256=digest(args.prepared_pilot/'dataset_manifest.json'))
    if args.resume and (not args.run_name or args.smoke):
        raise ValueError('--resume requires --run-name and cannot be used with --smoke')
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in ('.','..')):
        raise ValueError('--run-name must be a simple directory name, not a path')
    stamp = args.run_name or time.strftime('%Y%m%d_%H%M%S') + f'_{time.time_ns()%1000000:06d}'
    category = 'checks' if args.smoke or args.prepare_only else 'runs'
    directory = args.output/category/stamp
    directory.mkdir(parents=True, exist_ok=args.resume)
    # Keep provenance inside each model's config/checkpoint, not source copies.
    args.input_provenance = inputs
    if args.resume and (directory/'inputs.json').exists():
        # Compatibility with the old output layout.
        old=json.loads((directory/'inputs.json').read_text(encoding='utf-8'))
        if old!=inputs:
            raise ValueError('Input files changed; cannot resume')
    coverage = dict(context=data.context, graph_count=len(archive.times),
                    missing_slots=archive.metadata['missing_slots'], splits={})
    used = set()
    for split, ts in data.splits.items():
        ids = archive.ids(ts[:, None] + np.arange(-data.context+1, 1))
        unique = np.unique(ids)
        used.update(unique[unique > 0].tolist())
        coverage['splits'][split] = dict(samples=len(ts), first_origin=int(ts[0]), last_origin=int(ts[-1]),
                                       graph_bank_ids=unique.tolist(),
                                       samples_without_graph_at_origin=int((ids[:, -1] == 0).sum()))
    coverage['unused_graph_bank_ids'] = sorted(set(range(1, len(archive.times)+1))-used)
    if coverage['unused_graph_bank_ids'] and not args.train_start_index:
        raise ValueError(f'Some archive graphs never align with this dataset: {coverage["unused_graph_bank_ids"]}')
    print(f'Validated {len(archive.times)} graphs; context={data.context}; output={directory}', flush=True)
    coverage['graph_availability'] = args.graph_availability
    coverage['last_graph_available'] = int(archive.times[-1]) if len(archive.times) else None
    coverage['last_training_origin'] = int(data.splits['train'][-1])
    write_json(directory/'coverage.json', coverage)
    if args.prepare_only:
        # Explicit data-audit mode still exports its requested preparation files.
        archive.save(directory/'graphs')
        write_json(directory/'inputs.json', inputs)
        write_json(directory/'coverage.json', coverage)
        return directory
    modes = ['adaptive', 'causal', 'combined'] if args.mode == 'all' else [args.mode]
    summary = []
    for mode in modes:
        summary.append(run_mode(args, data, archive, directory, mode, diagnostic_archive=diagnostic_archive))
    with open(directory/'comparison.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    return directory


if __name__ == '__main__':
    main()
