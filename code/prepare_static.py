"""Validate a prepared dataset and export or validate its static prefix prior."""
import json
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'code'))
from pilot_protocol import validate_pilot
from static_provenance import validate_static_provenance


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    a = p.parse_args()
    meta = validate_pilot(a.prepared, expected_protocol='frozen_train')
    config = json.loads((a.prepared/'graphs/graph_config.json').read_text(encoding='utf-8'))
    matrix, provenance = a.prepared/'static_prior.npz', a.prepared/'static_prior.json'
    if matrix.exists() != provenance.exists():
        raise ValueError('Static matrix/provenance pair is incomplete; preserve and inspect the existing file')
    if not matrix.exists():
        subprocess.run([sys.executable, str(ROOT/'export_prefix_prior.py'),
            '--raw', str(a.prepared/'cleaned.npz'), '--graphs', str(a.prepared/'graphs/per_window'),
            '--available-index', str(meta['static_available_index']), '--alpha', str(config['alpha']),
            '--output', str(matrix)], check=True)
    validate_static_provenance(Namespace(static_matrix=matrix, static_provenance=provenance,
        raw=a.prepared/'cleaned.npz', graphs=a.prepared/'graphs/per_window',
        alpha=config['alpha'], graph_weighting='confidence',
        static_available_index=meta['static_available_index'], train_start_index=meta['static_available_index']))
    print('Static prefix matrix and provenance validated:', matrix)


if __name__ == '__main__':
    main()
