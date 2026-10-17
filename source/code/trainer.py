"""Server training entry point. --prepare-only never trains; --smoke is bounded."""
import argparse
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
from torch.utils.data import DataLoader, Subset, RandomSampler

from data_pipeline import Samples, digest, load_prepared
from forecast_model import PaperGraphWaveNet
from metrics import Metrics, masked_mae
from graph_controls import graph_control
from mean_teacher import (unwrap, is_constant_buffer, make_teacher, update_teacher,
                          perturb_inputs, consistency_loss)
import pi_model

ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'Graph-WaveNet-master/data/data_full_year')  # train/val/test位置
    p.add_argument('--raw', type=Path, default=ROOT/'Graph-WaveNet-master/data/full_year_cleaned.npz')  # 原始NPZ位置
    p.add_argument('--graphs', type=Path, default=ROOT/'per_window')  # PCMCI图目录
    p.add_argument('--output', type=Path, default=ROOT/'newcode/outputs')  # 结果保存位置
    p.add_argument('--mode', choices=['causal', 'adaptive', 'combined', 'all'], default='all')
    p.add_argument('--variables', choices=['all', 'flow'], default='all')
    p.add_argument('--device', default='cpu')
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--architecture', choices=['paper'], default='paper')
    p.add_argument('--graph-control', choices=['dynamic','static','hybrid','shuffled','collapsed','local'], default='dynamic',
                   help='static: first available lagged graph; hybrid: static+dynamic with shared weights; local: self edges')
    p.add_argument('--label-ratio', type=float, default=1., help='Fraction of training forecast origins receiving supervision; historical observations remain available')
    p.add_argument('--semi-supervised', action='store_true', help='EMA teacher consistency on unsupervised training origins (not a missing-sensor protocol)')
    p.add_argument('--ssl-method', choices=['none', 'mean_teacher', 'pi_model'], default='none',
                   help='Explicit SSL baseline; --semi-supervised remains a compatibility alias')
    p.add_argument('--static-weight', type=float, default=0.5,
                   help='Hybrid only: weight of the earliest available fixed lagged PCMCI graph')
    p.add_argument('--ssl-weight', type=float, default=0.1)
    p.add_argument('--ssl-warmup', type=int, default=10)
    p.add_argument('--ema-decay', type=float, default=0.99)
    p.add_argument('--input-noise', type=float, default=0.02, help='Student input noise in normalized units; teacher sees clean histories')
    p.add_argument('--pi-weight', type=float, default=0.1, help='Pi-model normalized MSE weight; independent of Mean Teacher ssl-weight')
    p.add_argument('--pi-noise', type=float, default=0.02)
    p.add_argument('--pi-warmup', type=int, default=10)
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
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--cache-dir', type=Path, default=None, help='Prepared input cache; default OUTPUT/cache')
    p.add_argument('--no-data-cache', action='store_true', help='Read and validate original inputs on every launch')
    p.add_argument('--rebuild-cache', action='store_true', help='Force revalidation and replacement of prepared input cache')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--skip-test', action='store_true', help='Validation-only tuning: save best checkpoint without evaluating test data')
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
               ssl_method='none'):
    if teacher is not None and ssl_method == 'pi_model':
        raise ValueError('Pi-model must not use a Mean Teacher instance')
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
            if training and not bool(valid.any()) and teacher is None and ssl_method != 'pi_model':
                continue
            cpu_y, cpu_valid = y, valid
            x, y, valid, graph_ids = (v.to(device, non_blocking=True) for v in (x, y, valid, graph_ids))
            clean_x = x
            if training and teacher is not None:
                x = perturb_inputs(x, input_noise)
            elif training and ssl_method == 'pi_model':
                x = pi_model.perturb(x, input_noise)
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
        if ssl_method == 'pi_model':
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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=split == 'train' and sampler is None,
                      sampler=sampler,
                      num_workers=args.workers, drop_last=False,
                      persistent_workers=args.workers > 0,
                      pin_memory=torch.device(args.device).type == 'cuda',
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
    return loader.sampler.generator if isinstance(loader.sampler, RandomSampler) else loader.generator


def run_mode(args, data, archive, directory, mode):
    seed_all(args.seed)
    run = directory/mode
    run.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device(args.device)
    model = PaperGraphWaveNet(
        data.values.shape[1], data.horizon, data.flow_indices, archive.bank, mode,
        args.hidden, args.blocks, args.layers, args.dropout,
        static_weight=args.static_weight if args.graph_control == 'hybrid' and mode != 'adaptive' else 0.
    ).to(device)
    if device.type == 'cuda' and args.gpus > 1:
        first = device.index or 0
        ids = list(range(first, first+args.gpus))
        if ids[-1] >= torch.cuda.device_count():
            raise ValueError('Requested GPU range is unavailable')
        model = torch.nn.DataParallel(model, device_ids=ids, output_device=first)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                          patience=args.lr_patience, min_lr=1e-5)
    splits = ('train', 'val') if args.skip_test else ('train', 'val', 'test')
    loaders = {s: make_loader(data, archive, s, args) for s in splits}
    teacher = None
    if args.ssl_method == 'mean_teacher' or (args.semi_supervised and args.ssl_method == 'none'):
        teacher = make_teacher(model)
    mean = torch.as_tensor(data.output_mean, device=device)
    std = torch.as_tensor(data.output_std, device=device)
    config = dict(vars(args), mode=mode, context=data.context, horizon=data.horizon,
                  ssl_algorithm=('mean_teacher_regression' if teacher is not None else
                                 'pi_model_regression' if args.ssl_method == 'pi_model' else 'supervised'),
                  evaluation_model='student',
                  implementation='paper_pcmci_mean_teacher_20261017',
                  supervision_protocol='forecast_origin_subsampling_with_observed_histories',
                  supervised_origins=len(loaders['train'].label_origins),
                  supervision_mask_sha256=hashlib.sha256(loaders['train'].supervision.numpy().tobytes()).hexdigest(),
                  target_offsets=data.y_offsets.tolist(), frozen_adaptive_backbone=False,
                  code_hashes={name: digest(Path(__file__).parent/name) for name in
                               ('trainer.py', 'data_pipeline.py', 'forecast_model.py', 'graphwavenet.py',
                                'mean_teacher.py', 'pi_model.py', 'metrics.py', 'graph_controls.py')},
                  variables_count=data.values.shape[1], parameters=sum(p.numel() for p in model.parameters()),
                  receptive_field=int(unwrap(model).receptive_field),
                  temporal_kernel=int(getattr(unwrap(model), 'temporal_kernel', 2)),
                  python_version=platform.python_version(), cuda_version=torch.version.cuda,
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
                    'graph_control','static_weight','label_ratio','semi_supervised','ssl_method',
                    'ssl_weight','ssl_warmup','ema_decay','input_noise','pi_weight','pi_noise','pi_warmup'):
            if saved['config'].get(key) != config.get(key):
                raise ValueError(f'Resume configuration mismatch: {key}')
        for key, expected in [('scaler_mean', data.mean), ('scaler_std',data.std), ('graph_times',archive.times)]:
            if not np.array_equal(saved[key].cpu().numpy(),expected):
                raise ValueError(f'Resume data mismatch: {key}')
        unwrap(model).load_state_dict(saved['model'])
        if teacher is not None:
            teacher.load_state_dict(saved['teacher'])
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
        start = time.time()
        weight = args.ssl_weight*min(1., epoch/max(1, args.ssl_warmup)) if teacher is not None else 0.
        noise = args.input_noise
        if args.ssl_method == 'pi_model':
            weight = args.pi_weight*min(1., epoch/max(1, args.pi_warmup))
            noise = args.pi_noise
        train, _ = pass_epoch(model, loaders['train'], device, mean, std, optimizer,
                              teacher=teacher, ssl_weight=weight, input_noise=noise, ema_decay=args.ema_decay,
                              ssl_method=args.ssl_method)
        val, _ = pass_epoch(model, loaders['val'], device, mean, std)
        score = val['overall']['mae']
        if score is None or not np.isfinite(score):
            raise ValueError('Validation has no finite observed-target metric')
        row = dict(epoch=epoch, seconds=time.time()-start, learning_rate=optimizer.param_groups[0]['lr'],
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
    if args.semi_supervised and args.ssl_method == 'pi_model':
        raise ValueError('--semi-supervised is a Mean Teacher alias; do not combine it with pi_model')
    if args.semi_supervised:
        args.ssl_method = 'mean_teacher'
    args.semi_supervised = args.ssl_method == 'mean_teacher'
    if min(args.pi_weight, args.pi_noise, args.pi_warmup) < 0:
        raise ValueError('Pi-model weight, noise and warmup must be nonnegative')
    if not 0 <= args.static_weight <= 1:
        raise ValueError('--static-weight must be in [0, 1]')
    if min(args.epochs, args.batch_size, args.hidden, args.blocks, args.layers, args.context, args.threads, args.gpus) < 1:
        raise ValueError('Training dimensions and epochs must be positive')
    if min(args.patience,args.lr_patience,args.workers) < 0 or not 0 <= args.dropout < 1:
        raise ValueError('Invalid patience/workers/dropout')
    if not 0 < args.label_ratio <= 1 or not 0 <= args.ema_decay < 1 or min(args.ssl_weight, args.ssl_warmup, args.input_noise) < 0:
        raise ValueError('Invalid label budget or SSL configuration')
    if args.ssl_method != 'none' and args.label_ratio == 1:
        raise ValueError('SSL requires --label-ratio < 1; full-label runs use supervised learning')
    if args.mode == 'adaptive' and args.graph_control != 'dynamic':
        raise ValueError('Adaptive baseline does not use graphs; graph controls are inapplicable')
    torch.set_num_threads(args.threads)
    # Cache includes full alignment validation, graph preparation and input hashes.
    data, archive, inputs = load_prepared(args)
    archive = graph_control(archive, args.graph_control, args.seed)
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
    if coverage['unused_graph_bank_ids']:
        raise ValueError(f'Some archive graphs never align with this dataset: {coverage["unused_graph_bank_ids"]}')
    print(f'Validated {len(archive.times)} graphs; context={data.context}; output={directory}', flush=True)
    if args.prepare_only:
        # Explicit data-audit mode still exports its requested preparation files.
        archive.save(directory/'graphs')
        write_json(directory/'inputs.json', inputs)
        write_json(directory/'coverage.json', coverage)
        return directory
    modes = ['adaptive', 'causal', 'combined'] if args.mode == 'all' else [args.mode]
    summary = []
    for mode in modes:
        summary.append(run_mode(args, data, archive, directory, mode))
    with open(directory/'comparison.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    return directory


if __name__ == '__main__':
    main()
