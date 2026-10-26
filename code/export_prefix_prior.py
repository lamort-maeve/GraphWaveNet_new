"""Export an explicitly dated early-prefix signed spatial prior and provenance."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'code'))
import numpy as np
from data_pipeline import GraphArchive, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--graphs', type=Path, required=True)
    p.add_argument('--available-index', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--alpha', type=float, default=.05)
    p.add_argument('--trusted-object-metadata', action='store_true')
    a = p.parse_args()
    if a.output.suffix != '.npz' or a.output.exists() or a.output.with_suffix('.json').exists():
        p.error('Use a new .npz path; existing artifacts are never overwritten')
    with np.load(a.raw, allow_pickle=a.trusted_object_metadata) as z:
        graph = GraphArchive(a.graphs, z['node_ids'], z['feature_names'], a.alpha)
    used = [w for w in graph.metadata['windows'] if w['available'] <= a.available_index]
    if not used:
        p.error('No graphs available by requested prefix time')
    # Equal weighting over prefix snapshots and nonzero lags; lag identity is
    # deliberately lost only in this static spatial reference.
    matrix = graph.bank[1:len(used)+1, 1:].mean(axis=(0, 1))
    ids = [f'{n}:{f}' for n in graph.metadata['node_ids'] for f in graph.metadata['features']]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, adjacency=matrix, node_ids=np.asarray(ids))
    meta = dict(schema=1, available_index=a.available_index,
                source_end_exclusive=max(w['end'] for w in used),
                matrix_sha256=digest(a.output), raw_sha256=digest(a.raw),
                sources=[dict(file=w['file'], sha256=w['sha256']) for w in used],
                alpha=a.alpha, weighting='confidence', aggregation='mean_window_and_nonzero_lag')
    a.output.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(a.output)


if __name__ == '__main__':
    main()
