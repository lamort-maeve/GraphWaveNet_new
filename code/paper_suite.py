"""Revision experiment matrix. Validation-only and command preview by default."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def matrix(suite):
    common = ['--mode', 'combined', '--graph-kernel', '3', '--history-padding', 'masked',
              '--lag-normalization', 'valid', '--history-init', 'uniform']
    rows = []
    if suite == 'history':
        for k in (1, 2, 3, 5):
            for fusion in (('learned', 'uniform') if k > 1 else ('learned',)):
                rows.append((f'k{k}_{fusion}', common+['--graph-kernel', str(k), '--history-fusion', fusion]))
        for control in ('zero', 'shuffled', 'unsigned'):
            rows.append((control, common+['--graph-control', control]))
        rows += [('k3_zero_padding', common+['--history-padding', 'zero']),
                 ('k3_all_lags', common+['--lag-normalization', 'all']),
                 ('adaptive', ['--mode', 'adaptive']),
                 ('causal_k3', common+['--mode', 'causal'])]
        return [(name, opts+['--label-ratio', '.25']) for name, opts in rows]
    if suite == 'ssl':
        for ratio in (.1, .25, .5):
            for method in ('none', 'mean_teacher', 'pi_model', 'temporal_ensembling', 'self_training'):
                rows.append((f'label{int(ratio*100)}_{method}', common+['--label-ratio', str(ratio), '--ssl-method', method]))
        rows.append(('full_supervised', common))
    elif suite == 'mechanism':
        common += ['--diagnostic-fraction', '.1']
        rows.append(('mt', common+['--label-ratio', '.25', '--ssl-method', 'mean_teacher', '--diagnose-teacher']))
    elif suite == 'static':
        # Caller supplies common dated prior and common start index. The weight
        # zero row retains the same static parameters but does not use messages.
        for weight in (0., .5, 1.):
            for control in ('real', 'random'):
                rows.append((f'static{weight}_{control}', common+['--label-ratio', '.25',
                             '--graph-control', 'hybrid', '--static-weight', str(weight), '--static-control', control]))
    elif suite == 'full':
        common += ['--graph-control', 'hybrid', '--static-weight', '.5']
        for ratio in (.1, .25, .5):
            for method in ('none', 'mean_teacher', 'pi_model', 'temporal_ensembling', 'self_training'):
                rows.append((f'full_label{int(ratio*100)}_{method}', common+[
                    '--label-ratio', str(ratio), '--ssl-method', method]))
        rows.append(('full_label100_supervised', common))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['history', 'ssl', 'mechanism', 'static', 'full'], required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    p.add_argument('--prefix', default='revision1')
    p.add_argument('--only', nargs='+')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--report-test', action='store_true', help='Enable only after method/hyperparameter choices are frozen')
    p.add_argument('--plan', type=Path, help='Save exact commands as JSON')
    a, shared = p.parse_known_args()
    if len(set(a.seeds)) != len(a.seeds) or not a.seeds:
        p.error('Seeds must be unique')
    owned = {'--mode', '--graph-kernel', '--history-padding', '--lag-normalization', '--history-fusion', '--history-init',
             '--graph-control', '--label-ratio', '--ssl-method', '--semi-supervised', '--seed', '--run-name',
             '--static-control', '--static-weight', '--skip-test', '--resume', '--final-test'}
    if any(v.split('=')[0] in owned for v in shared):
        p.error('Suite owns architecture, ablation, label, seed and test-selection switches')
    rows = matrix(a.suite)
    if a.only:
        if set(a.only)-{name for name, _ in rows}:
            p.error('Unknown --only name')
        rows = [(n, v) for n, v in rows if n in a.only]
    commands = []
    for seed in a.seeds:
        for name, opts in rows:
            cmd = [sys.executable, str(Path(__file__).with_name('train.py')), '--epochs', '100',
                   '--patience', '0', '--batch-size', '16', *shared, *opts, '--seed', str(seed),
                   '--run-name', f'{a.prefix}_{a.suite}_{name}_s{seed}']
            if not a.report_test:
                cmd.append('--skip-test')
            else:
                cmd.append('--final-test')
            commands.append(cmd)
    if a.plan:
        a.plan.parent.mkdir(parents=True, exist_ok=True)
        if a.plan.exists() and json.loads(a.plan.read_text(encoding='utf-8')) != commands:
            p.error('Plan already exists with different commands; use a new prefix or plan path')
        a.plan.write_text(json.dumps(commands, indent=2, ensure_ascii=False), encoding='utf-8')
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
        if a.execute:
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
