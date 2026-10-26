"""Paired SSL/graph experiments; preview and reuse completed validation runs."""
import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INDEX = ROOT/'已完成实验索引.json'
METHODS = {'SUP': 'none', 'MT': 'mean_teacher', 'PI': 'pi_model',
           'TE': 'temporal_ensembling', 'ST': 'self_training'}
METHOD_NAMES = {'SUP': '纯监督', 'MT': '原版 MeanTeacher', 'PI': 'Pi-model',
                'TE': 'Temporal Ensembling', 'ST': 'Self-training'}
GRAPHS = {'J': '无静态/动态因果消息', 'A': '仅动态因果消息',
          'K': '仅静态因果消息', 'G': '静态＋动态因果消息'}
ROWS = {f'{m}_{g}': (m, g, {}) for m in METHODS for g in GRAPHS}
ROWS.update({
    'M': ('SUP', 'G', {'static-control': 'random'}),
    'N': ('SUP', 'G', {'lag-readout': 'synchronous'}),
    'O': ('SUP', 'G', {'history-weight-sharing': 'shared'}),
    'P': ('SUP', 'G', {'history-fusion': 'uniform'}),
    'Q': ('SUP', 'G', {'graph-kernel': '1', 'history-fusion': 'learned'}),
    'T': ('SUP', 'A', {'lag-readout': 'synchronous'}),
})
MECHANISM_NAMES = {'M': '随机静态参考', 'N': '同步读取＋静态参考',
                   'O': '跨 lag 共享历史权重', 'P': '均匀历史＋静态参考',
                   'Q': '单快照＋静态参考', 'T': '同步读取、无静态参考'}


def description(row):
    method, graph, _ = ROWS[row]
    return MECHANISM_NAMES.get(row, METHOD_NAMES[method]+'；'+GRAPHS[graph])


def options(row, seed, prepared, output, device='cuda:0'):
    method, graph, overrides = ROWS[row]
    p = str(prepared).replace('\\', '/').rstrip('/')
    opts = {
        'epochs': '100', 'patience': '10', 'batch-size': '16',
        'prepared-pilot': p, 'context': '25', 'train-start-index': '4032',
        'output': str(output), 'graph-availability': 'frozen_train', 'mode': 'combined',
        'graph-control': 'dynamic', 'graph-kernel': '3', 'history-padding': 'masked',
        'lag-normalization': 'valid', 'history-init': 'uniform', 'causal-revision': 'selective',
        'adaptive-scale': '1', 'label-ratio': '0.25', 'seed': str(seed), 'device': device,
        'threads': '7', 'workers': '0', 'causal-execution': 'reference', 'batch-fetch': 'vectorized',
        'history-fusion': 'shrunk', 'history-uniform-mix': '0.5', 'ssl-method': METHODS[method],
        'static-fusion': 'convex', 'static-weight': '0', 'lag-readout': 'aligned',
        'history-weight-sharing': 'per_lag', 'dynamic-message-scale': '0' if graph in ('J', 'K') else '1',
        'hidden': '32', 'blocks': '4', 'layers': '2', 'dropout': '0.3',
        'learning-rate': '0.001', 'weight-decay': '0.0001', 'lr-patience': '5',
        'graph-weighting': 'confidence', 'alpha': '0.05', 'variables': 'all',
    }
    if graph in ('K', 'G'):
        opts.update({'graph-control': 'hybrid', 'static-matrix': p+'/static_prior.npz',
            'static-provenance': p+'/static_prior.json', 'static-available-index': '4032',
            'static-fusion': 'gated_residual', 'static-weight': '0.5', 'static-control': 'real'})
    if method == 'MT':
        opts.update({'ssl-weight': '0.1', 'ssl-warmup': '10', 'ema-decay': '0.99', 'input-noise': '0.02'})
    elif method == 'PI':
        opts.update({'pi-weight': '0.1', 'pi-warmup': '10', 'pi-noise': '0.02'})
    elif method == 'TE':
        opts.update({'te-weight': '0.1', 'te-warmup': '10', 'te-noise': '0.02', 'te-decay': '0.6'})
    elif method == 'ST':
        opts.update({'st-weight': '0.1', 'st-warmup': '10', 'st-noise': '0.02',
                     'st-pretrain': '10', 'st-refresh': '5'})
    opts.update(overrides)
    opts['run-name'] = f'general_{row}_s{seed}'
    return opts


def command(row, seed, prepared, output, code_root=ROOT, device='cuda:0', python=sys.executable):
    opts = options(row, seed, prepared, output, device)
    args = [python, str(code_root).replace('\\', '/').rstrip('/')+'/train.py']
    for key, value in opts.items():
        args.extend(['--'+key, str(value).replace('\\', '/')])
    if ROWS[row][0] == 'MT':
        args.append('--ssl-clean-labeled')
    args.append('--skip-test')
    return args


def source_hashes():
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT/'train.py', *sorted((ROOT/'code').glob('*.py'))]}


def scan_results(root):
    """Require best checkpoint, selected epoch, history and saved label mask."""
    entries = []
    for path in sorted(Path(root).rglob('config.json')):
        run = path.parent
        try:
            config = json.loads(path.read_text(encoding='utf-8'))
            selection = json.loads((run/'validation_selection.json').read_text(encoding='utf-8'))
            if config.get('smoke') or selection.get('smoke_only'):
                continue
            if not (run/'best.pt').is_file() or (run/'best.pt').stat().st_size == 0:
                continue
            with (run/'history.csv').open(encoding='utf-8', newline='') as stream:
                history = list(csv.DictReader(stream))
            chosen = [r for r in history if int(r['epoch']) == selection['best_epoch']]
            if len(chosen) != 1 or not math.isfinite(float(chosen[0]['val_mae'])):
                continue
            if not math.isclose(float(chosen[0]['val_mae']), selection['best_validation_mae'], rel_tol=1e-7):
                continue
            hashes = {name: hashlib.sha256((run/name).read_bytes()).hexdigest()
                      for name in ('config.json', 'validation_selection.json', 'history.csv', 'labeled_origins.npy')}
            entries.append({'run': str(run.resolve()), 'config': config, 'selection': selection,
                            'evidence_sha256': hashes})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return entries


def load_completed(results=None, output=None):
    entries = []
    if results is None and INDEX.exists():
        entries.extend(json.loads(INDEX.read_text(encoding='utf-8'))['completed'])
    if results is not None:
        if not Path(results).is_dir():
            raise ValueError('Results directory does not exist: '+str(results))
        entries.extend(scan_results(results))
    if output is not None and (Path(output)/'runs').is_dir():
        entries.extend(scan_results(Path(output)/'runs'))
    return entries


def completed_match(row, seed, prepared, entries):
    opts = options(row, seed, prepared, '')
    expected = {k.replace('-', '_'): v for k, v in opts.items()
                if k not in ('output', 'run-name', 'device', 'threads', 'workers')}
    expected['ssl_clean_labeled'] = ROWS[row][0] == 'MT'
    expected.update({'diagnostic_fraction': 0., 'horizon': 12, 'architecture': 'paper',
                     'supervision_protocol': 'forecast_origin_subsampling_with_observed_histories'})
    for entry in entries:
        config = dict(entry['config'])
        # v4 had these exact defaults before exposing their CLI switches.
        if config.get('implementation') == 'paper_lag_structural_v4_20261025':
            for key, default in {'lag_readout': 'aligned', 'history_weight_sharing': 'per_lag',
                                 'dynamic_message_scale': 1.}.items():
                config.setdefault(key, default)
        # The dynamic-only architecture never uses the static-weight argument.
        if config.get('graph_control') == 'dynamic' and not config.get('static_matrix'):
            config['static_weight'] = 0.
        same = True
        for key, value in expected.items():
            actual = config.get(key)
            if isinstance(value, bool):
                equal = actual is value
            elif isinstance(actual, (float, int)):
                try:
                    equal = float(actual) == float(value)
                except (TypeError, ValueError):
                    equal = False
            else:
                equal = str(actual).replace('\\', '/') == str(value).replace('\\', '/')
            if not equal:
                same = False
                break
        if same:
            return entry
    return None


def selected_rows(phase):
    if phase == 'main':
        return [k for k in ROWS if '_' in k]
    if phase == 'mt':
        return [k for k in ROWS if k.startswith('MT_')]
    if phase == 'ssl':
        return [k for k in ROWS if k.startswith(('PI_', 'TE_', 'ST_'))]
    if phase == 'mechanism':
        return list(MECHANISM_NAMES)
    return list(ROWS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['main', 'mt', 'ssl', 'mechanism', 'all'], default='main')
    parser.add_argument('--only', nargs='+', choices=list(ROWS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--results', type=Path, help='Rescan actual results instead of the bundled V4 index')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--plan', type=Path)
    a = parser.parse_args()
    if len(set(a.seeds)) != len(a.seeds):
        parser.error('Seeds must be unique')
    entries = load_completed(a.results, a.output)
    commands, reused = [], []
    for seed in a.seeds:
        for row in a.only or selected_rows(a.phase):
            match = completed_match(row, seed, a.prepared, entries)
            if match:
                reused.append({'row': row, 'seed': seed, 'run': match['run']})
            else:
                commands.append(command(row, seed, a.prepared, a.output, device=a.device))
    plan = {'scope': 'validation only; paired SSL/graph study', 'commands': commands,
            'reused': reused, 'source_sha256': source_hashes()}
    if a.plan:
        a.plan.parent.mkdir(parents=True, exist_ok=True)
        if a.plan.exists() and json.loads(a.plan.read_text(encoding='utf-8')) != plan:
            parser.error('Existing plan differs; use a new plan file')
        a.plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    for item in reused:
        print('# REUSE '+item['row']+f" seed={item['seed']} "+item['run'], flush=True)
    for args in commands:
        print(shlex.join(args), flush=True)
    print(f'# pending={len(commands)} reused={len(reused)}', flush=True)
    if a.execute:
        for args in commands:
            run = a.output/'runs'/args[args.index('--run-name')+1]
            if run.exists():
                parser.error(f'Incomplete or incompatible run exists: {run}; inspect it before restarting')
        for args in commands:
            subprocess.run(args, check=True)


if __name__ == '__main__':
    main()
