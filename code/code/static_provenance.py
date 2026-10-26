"""Reject undocumented external priors. Metadata is a claim, hashes verify files."""
import json
from data_pipeline import digest


def validate_static_provenance(args):
    if args.static_provenance is None:
        raise ValueError('External prior needs --static-provenance; use export_prefix_prior.py')
    meta = json.loads(args.static_provenance.read_text(encoding='utf-8'))
    if meta.get('matrix_sha256') != digest(args.static_matrix):
        raise ValueError('Static prior hash mismatch')
    if meta.get('raw_sha256') != digest(args.raw):
        raise ValueError('Static prior was not exported against this raw dataset')
    if meta.get('alpha') != args.alpha or meta.get('weighting') != args.graph_weighting:
        raise ValueError('Static and dynamic graph threshold/weighting must match the declared comparison')
    available = meta['available_index']
    if available != args.static_available_index or not 0 <= meta['source_end_exclusive'] <= available:
        raise ValueError('Static prior availability contradicts provenance')
    if args.train_start_index < available:
        raise ValueError('Use a common --train-start-index >= static availability in matched comparisons')
    sources = meta.get('sources', [])
    if not sources:
        raise ValueError('Static prior requires source graph hashes')
    for source in sources:
        path = args.graphs/source['file']
        if path.parent.resolve() != args.graphs.resolve() or digest(path) != source['sha256']:
            raise ValueError('Static source graph path/hash mismatch')
        import numpy as np
        with np.load(path, allow_pickle=False) as z:
            if int(z['window_end_index']) > meta['source_end_exclusive'] or int(z['graph_time_index']) > available:
                raise ValueError('Static prior uses a future graph')
