"""Server training entry point. --prepare-only never trains; --smoke is bounded."""
import argparse
import csv
import json
import inspect
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler

from causal_data import GraphArchive, TrafficData, Samples, digest, load_prepared
from causal_model import DynamicGraphWaveNet
from causal_model_v1 import DynamicGraphWaveNet as V1GraphWaveNet

ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'Graph-WaveNet-master/data/data_full_year')"""修改train，val，test文件夹的位置"""
    p.add_argument('--raw', type=Path, default=ROOT/'Graph-WaveNet-master/data/full_year_cleaned.npz')"""npz文件的位置"""
    p.add_argument('--graphs', type=Path, default=ROOT/'per_window')"""per_window文件夹的位置"""
    p.add_argument('--output', type=Path, default=ROOT/'newcode/outputs')"""输出结果的保存位置"""
    p.add_argument('--mode', choices=['causal', 'adaptive', 'combined', 'all'], default='all')
    p.add_argument('--variables', choices=['all', 'flow'], default='all')
    p.add_argument('--device', default='cpu')
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--architecture', choices=['v1','v2'], default='v2')
    p.add_argument('--graph-operator', choices=['temporal_tensor','gated_lag','v1'], default='temporal_tensor')
    p.add_argument('--graph-kernel', type=int, default=3,
                   help='Number of available graph snapshots in the causal temporal convolution')
    p.add_argument('--graph-history-fusion', choices=['signed_convex', 'signed_conv'], default='signed_convex',
                   help='Separate signs before convex temporal convolution, or use the previous unconstrained convolution')
    p.add_argument('--no-multiscale', action='store_true', help='Disable the node-local multiscale temporal module')
    p.add_argument('--graph-weighting', choices=['confidence', 'legacy'], default='confidence',
                   help='Preserve PCMCI effect strength and discount near-threshold q values')
    p.add_argument('--no-pcmci-refinement', action='store_true', help='Disable PCMCI edge/lag selection in stacked graph layers; readout is controlled separately')
    p.add_argument('--no-causal-readout', action='store_true', help='Disable the shallow single-hop PCMCI forecast branch')
    p.add_argument('--no-persistence', action='store_true')
    p.add_argument('--no-variable-embedding', action='store_true')
    p.add_argument('--trusted-object-metadata', action='store_true', help='Allow object dtype metadata in your own trusted NPZ')
    p.add_argument('--resume', action='store_true', help='Explicitly resume an existing matching V2 run')
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


def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


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


class Metrics:
    def __init__(self, horizon):
        self.sums = torch.zeros((horizon, 5), dtype=torch.float64)

    @torch.no_grad()
    def update(self, pred, truth, valid):
        # True zero flow is a valid target; only MAPE excludes zero denominators.
        diff = torch.where(valid, pred-truth, torch.zeros_like(pred)).double()
        pct = valid & (truth.abs() > 1e-5)
        vals = [diff.abs().sum((0, 2)), diff.square().sum((0, 2)), valid.sum((0, 2)),
                torch.where(pct, diff.abs()/truth.abs().clamp_min(1e-5), torch.zeros_like(diff)).sum((0, 2)),
                pct.sum((0, 2))]
        # Accumulate on device; one CPU transfer per epoch, not per batch.
        self.sums = self.sums.to(diff.device)
        self.sums += torch.stack(vals, -1).detach()

    @staticmethod
    def row(s):
        a, q, count, pct, pct_count = map(float, s)
        return dict(mae=a/count if count else None, rmse=(q/count)**0.5 if count else None,
                    mape=pct/pct_count if pct_count else None, valid_count=int(count),
                    mape_count=int(pct_count))

    def result(self):
        sums = self.sums.cpu().numpy()
        return dict(overall=self.row(sums.sum(0)),
                    by_horizon=[dict(horizon=i+1, **self.row(s)) for i, s in enumerate(sums)])


def masked_mae(pred, truth, valid):
    # Boolean indexing creates a dynamic-sized CUDA output (and a host sync).
    # A fixed-shape reduction has the same objective, including true zero targets.
    return torch.where(valid, (pred-truth).abs(), torch.zeros_like(pred)).sum() / valid.sum().clamp_min(1)


def pass_epoch(model, loader, device, mean, std, optimizer=None, capture=False):
    training = optimizer is not None
    model.train(training)
    metric = Metrics(getattr(loader, 'forecast_horizon', None) or len(loader.dataset[0][1]))
    predictions, targets, masks, origins = [], [], [], []
    finite = torch.ones((), dtype=torch.bool, device=device)
    parameters = list(model.parameters())
    # Single-GPU V2 can deduplicate the already-CPU IDs before the transfer.
    # CUDA unique has a dynamically sized result and otherwise synchronizes.
    cpu_graph_lookup = (device.type == 'cuda' and not isinstance(model, torch.nn.DataParallel)
                        and getattr(model, 'graph_operator', None) == 'temporal_tensor'
                        and getattr(model, 'mode', None) != 'adaptive')
    with torch.set_grad_enabled(training):
        for x, y, valid, graph_ids, origin in loader:
            has_targets = bool(valid.any()) if training else True
            if not has_targets:
                continue
            cpu_y, cpu_valid = y, valid
            graph_lookup = None
            if cpu_graph_lookup:
                unique, inverse = np.unique(graph_ids.numpy(), return_inverse=True)
                graph_lookup = tuple(torch.from_numpy(a.copy()).pin_memory().to(device, non_blocking=True)
                                     for a in (unique, inverse.reshape(-1)))
            x, y, valid, graph_ids = (v.to(device, non_blocking=True) for v in (x, y, valid, graph_ids))
            output = model(x, graph_ids, graph_lookup=graph_lookup) if graph_lookup is not None else model(x, graph_ids)
            pred = output * std + mean
            finite.logical_and_(torch.isfinite(pred).all())
            if training:
                loss = masked_mae(pred, y, valid)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Do not convert the CUDA norm to a Python bool every batch.
                norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=False)
                finite.logical_and_(torch.isfinite(norm))
                optimizer.step()
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
    return metric.result(), packed


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
    return loader


def loader_rng(loader):
    return loader.sampler.generator if isinstance(loader.sampler, RandomSampler) else loader.generator


def run_mode(args, data, archive, directory, mode):
    seed_all(args.seed)
    run = directory/mode
    run.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device(args.device)
    model_class = DynamicGraphWaveNet if args.architecture == 'v2' else V1GraphWaveNet
    extras = dict(context=data.context, graph_operator=args.graph_operator,
                  graph_kernel=args.graph_kernel,
                  multiscale=not args.no_multiscale, graph_history_fusion=args.graph_history_fusion,
                  pcmci_refinement=not args.no_pcmci_refinement, causal_readout=not args.no_causal_readout,
                  persistence=not args.no_persistence, variable_embedding=not args.no_variable_embedding) if args.architecture=='v2' else {}
    model = model_class(data.values.shape[1], data.horizon, data.flow_indices,
                       archive.bank, mode, args.hidden, args.blocks, args.layers, args.dropout, **extras).to(device)
    if device.type == 'cuda' and args.gpus > 1:
        first = device.index or 0
        ids = list(range(first, first+args.gpus))
        if ids[-1] >= torch.cuda.device_count():
            raise ValueError('Requested GPU range is unavailable')
        model = torch.nn.DataParallel(model, device_ids=ids, output_device=first)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                          patience=args.lr_patience, min_lr=1e-5)
    loaders = {s: make_loader(data, archive, s, args) for s in ('train', 'val', 'test')}
    mean = torch.as_tensor(data.output_mean, device=device)
    std = torch.as_tensor(data.output_std, device=device)
    config = dict(vars(args), mode=mode, context=data.context, horizon=data.horizon,
                  implementation='pcmci_conditioned_20261011',
                  code_hashes={name: digest(Path(__file__).parent/name) for name in
                               ('train_dynamic.py', 'causal_data.py', 'causal_model.py', 'causal_model_v1.py')},
                  variables_count=data.values.shape[1], parameters=sum(p.numel() for p in model.parameters()),
                  receptive_field=int(unwrap(model).receptive_field),
                  temporal_kernel=int(getattr(unwrap(model), 'temporal_kernel', 2)),
                  torch_version=str(torch.__version__), numpy_version=np.__version__)
    config = json.loads(json.dumps(config, default=str))
    best, stale, history = float('inf'), 0, []
    start_epoch = 1
    if args.resume and (run/'last.pt').exists():
        saved = load_checkpoint(run/'last.pt','cpu')
        for key in ('code_hashes', 'input_provenance'):
            if key in saved['config'] and saved['config'][key] != config.get(key):
                raise ValueError(f'Resume provenance mismatch: {key}; start a new run')
        for key in ('architecture','variables','context','hidden','blocks','layers','dropout','graph_operator',
                    'graph_kernel','graph_history_fusion','graph_weighting','no_pcmci_refinement','no_causal_readout',
                    'no_multiscale','no_persistence','no_variable_embedding','seed','alpha','batch_size','learning_rate','weight_decay','lr_patience'):
            if saved['config'].get(key) != config.get(key):
                raise ValueError(f'Resume configuration mismatch: {key}')
        for key, expected in [('scaler_mean', data.mean), ('scaler_std',data.std), ('graph_times',archive.times)]:
            if not np.array_equal(saved[key].cpu().numpy(),expected):
                raise ValueError(f'Resume data mismatch: {key}')
        unwrap(model).load_state_dict(saved['model'])
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
                        if name in ('graph_bank', 'flow_indices')}
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
        train, _ = pass_epoch(model, loaders['train'], device, mean, std, optimizer)
        val, _ = pass_epoch(model, loaders['val'], device, mean, std)
        score = val['overall']['mae']
        if score is None or not np.isfinite(score):
            raise ValueError('Validation has no finite observed-target metric')
        row = dict(epoch=epoch, seconds=time.time()-start, learning_rate=optimizer.param_groups[0]['lr'],
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
        save_start = time.perf_counter()
        atomic_save(checkpoint, run/'last.pt')
        if improved:
            atomic_copy(run/'last.pt', run/'best.pt')
        print(f'{mode} epoch={epoch} train_MAE={train["overall"]["mae"]:.4f} '
              f'val_MAE={score:.4f} train+val_seconds={row["seconds"]:.1f} '
              f'save_seconds={time.perf_counter()-save_start:.1f}', flush=True)
        if args.patience and stale >= args.patience:
            break
    # Evaluation needs model weights only; do not upload the saved Adam state
    # and a second graph bank to CUDA just to immediately discard them.
    checkpoint = load_checkpoint(run/'best.pt', 'cpu')
    unwrap(model).load_state_dict(checkpoint['model'])
    test, arrays = pass_epoch(model, loaders['test'], device, mean, std, capture=True)
    test['best_epoch'] = checkpoint['epoch']
    test['best_validation_mae'] = checkpoint['validation']['overall']['mae']
    test['smoke_only'] = args.smoke
    test['target_offsets'] = data.y_offsets.tolist()
    write_json(run/'test_metrics.json', test)
    np.savez_compressed(run/'test_predictions.npz', **arrays)
    return dict(mode=mode, best_epoch=checkpoint['epoch'],
                best_validation_mae=test['best_validation_mae'], **test['overall'], smoke_only=args.smoke)


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.epochs, args.batch_size, args.hidden, args.blocks, args.layers, args.context, args.threads, args.gpus, args.graph_kernel) < 1:
        raise ValueError('Training dimensions and epochs must be positive')
    if min(args.patience,args.lr_patience,args.workers) < 0 or not 0 <= args.dropout < 1:
        raise ValueError('Invalid patience/workers/dropout')
    torch.set_num_threads(args.threads)
    # Cache includes full alignment validation, graph preparation and input hashes.
    data, archive, inputs = load_prepared(args)
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
    modes = ['causal', 'adaptive', 'combined'] if args.mode == 'all' else [args.mode]
    summary = [run_mode(args, data, archive, directory, mode) for mode in modes]
    with open(directory/'comparison.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    return directory


if __name__ == '__main__':
    main()
