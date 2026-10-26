"""Strict seed summaries or paired time-block MAE bootstrap; never select on test."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def paired_blocks(left, right, block_steps=2016, draws=5000, seed=42):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        for key in ('origin_index', 'target', 'observed'):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f'Paired prediction mismatch: {key}')
        valid, origin = a['observed'].astype(bool), a['origin_index']
        if len(np.unique(origin)) != len(origin) or np.any(np.diff(origin) <= 0):
            raise ValueError('Origins must be unique and chronological')
        errors = []
        for prediction in (a['prediction'], b['prediction']):
            if prediction.shape != valid.shape or not np.isfinite(prediction).all():
                raise ValueError('Invalid predictions')
            errors.append(np.where(valid, np.abs(prediction-a['target']), 0).sum(axis=(1, 2)))
        numerator = errors[0]-errors[1]
        count = valid.sum(axis=(1, 2))
    labels = origin//block_steps
    blocks = np.unique(labels)
    sums = np.asarray([[numerator[labels == k].sum(), count[labels == k].sum()] for k in blocks])
    sums = sums[sums[:, 1] > 0]
    if len(sums) < 3:
        raise ValueError('Need at least three observed time blocks')
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for i in range(draws):
        sample = sums[rng.integers(len(sums), size=len(sums))].sum(0)
        boot[i] = sample[0]/sample[1]
    return dict(mae_left_minus_right=float(sums[:, 0].sum()/sums[:, 1].sum()),
                ci95=np.quantile(boot, [.025, .975]).tolist(), blocks=len(sums), block_steps=block_steps,
                draws=draws, seed=seed, limitation='Conditional on these fitted runs; not seed uncertainty or an iid significance test.')


def summarize(runs, expected_seeds):
    rows, configs, horizon_rows, efficiency = [], [], [], []
    for run in runs:
        config = json.loads((run/'config.json').read_text(encoding='utf-8'))
        metrics = json.loads((run/'test_metrics.json').read_text(encoding='utf-8'))
        if metrics['smoke_only'] or config.get('smoke'):
            raise ValueError('Smoke results cannot enter a paper table')
        configs.append(config)
        rows.append(dict(seed=config['seed'], **metrics['overall']))
        horizon_rows.append(metrics['by_horizon'])
        with (run/'history.csv').open(encoding='utf-8', newline='') as stream:
            epochs = list(csv.DictReader(stream))
        efficiency.append(dict(seed=config['seed'], epochs=len(epochs),
            train_validation_seconds=sum(float(r['seconds']) for r in epochs),
            optimizer_steps=sum(int(r['optimizer_steps']) for r in epochs),
            peak_cuda_allocated_bytes=max(int(r.get('peak_cuda_allocated_bytes', 0)) for r in epochs),
            parameters=config['parameters'], trainable_parameters=config.get('trainable_parameters')))
    # Explicit run paths must be repeated seeds of ONE configuration.
    ignored = {'seed', 'run_name', 'output', 'resume', 'device', 'workers', 'threads', 'cache_dir',
               'supervision_mask_sha256', 'diagnose_teacher'}
    canonical = [{k: v for k, v in c.items() if k not in ignored} for c in configs]
    if any(c != canonical[0] for c in canonical[1:]):
        raise ValueError('Configurations/provenance differ beyond seed and runtime paths')
    seeds = [r['seed'] for r in rows]
    if len(set(seeds)) != len(seeds) or set(seeds) != set(expected_seeds):
        raise ValueError('Missing, unexpected or duplicate seeds')
    result = dict(seeds=seeds, runs=[str(p) for p in runs], metrics={})
    for key in ('mae', 'rmse', 'mape'):
        values = np.asarray([r[key] for r in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'Invalid {key}')
        result['metrics'][key] = dict(mean=float(values.mean()), std=float(values.std(ddof=1)), n=len(values))
    result['mape_unit'] = 'fraction; multiply by 100 for percent'
    result['by_horizon'] = []
    for h, offset in enumerate(configs[0]['target_offsets']):
        row = dict(target_offset=offset)
        for key in ('mae', 'rmse', 'mape'):
            values = np.array([r[h][key] for r in horizon_rows], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f'Undefined {key} at target offset {offset}')
            row[key] = dict(mean=float(values.mean()), std=float(values.std(ddof=1)))
        result['by_horizon'].append(row)
    result['efficiency_by_seed'] = efficiency
    result['timing_scope'] = 'Training plus validation; excludes preprocessing, PCMCI discovery, checkpoint writing and final test'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path)
    p.add_argument('--seeds', nargs='+', type=int, default=[42,43,44,45,46])
    p.add_argument('--paired', nargs=2, type=Path)
    p.add_argument('--block-steps', type=int, default=2016)
    p.add_argument('--draws', type=int, default=5000)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if bool(a.runs) == bool(a.paired) or a.block_steps < 1 or a.draws < 100 or len(a.seeds) < 2:
        p.error('Choose --runs or --paired, positive blocks, >=100 draws and >=2 summary seeds')
    result = summarize(a.runs, a.seeds) if a.runs else paired_blocks(*a.paired, a.block_steps, a.draws)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
