"""Validate prepared-data provenance and a complete rolling graph schedule."""
import json
from pathlib import Path
import numpy as np
from data_pipeline import digest
from graph_protocol import graph_schedule


def validate_pilot(root, require_graphs=True, expected_protocol=None):
    root = Path(root)
    meta = json.loads((root/'dataset_manifest.json').read_text(encoding='utf-8'))
    starts, ends, expected = graph_schedule(meta)
    if expected_protocol is not None and meta['graph_protocol'] != expected_protocol:
        raise ValueError('Dataset and experiment graph protocols differ; do not mix frozen and online runs')
    if digest(root/'dynamic_pcmci_window_manifest.npz') != meta['schedule_sha256']:
        raise ValueError('Graph schedule hash mismatch')
    with np.load(root/'dynamic_pcmci_window_manifest.npz', allow_pickle=False) as schedule:
        for key, values in (('window_start_indices',starts), ('window_end_indices',ends),
                            ('graph_time_indices',expected)):
            if not np.array_equal(schedule[key],values):
                raise ValueError(f'Graph schedule contents differ from protocol: {key}')
    if digest(root/'cleaned.npz') != meta['raw_sha256']:
        raise ValueError('Prepared raw file differs from dataset manifest')
    for split in ('train','val','test'):
        if digest(root/'forecast'/(split+'.npz')) != meta['split_sha256'][split]:
            raise ValueError(f'{split} hash mismatch')
    if not require_graphs:
        return meta
    config = json.loads((root/'graphs/graph_config.json').read_text(encoding='utf-8'))
    if config.get('protocol') != meta['graph_protocol'] or config.get('discovery_cutoff') != meta['discovery_cutoff']:
        raise ValueError('Graph discovery protocol/cutoff mismatch')
    if config['raw_sha256'] != meta['raw_sha256'] or config['dataset_manifest_sha256'] != digest(root/'dataset_manifest.json'):
        raise ValueError('Graphs were not constructed from this dataset/protocol')
    paths = sorted((root/'graphs/per_window').glob('graph_*.npz'))
    if [p.name for p in paths] != [f'graph_{i:04d}.npz' for i in range(len(expected))]:
        raise ValueError(f'Incomplete graph bank: need {len(expected)} exact schedule files, found {len(paths)}')
    with np.load(root/'cleaned.npz') as raw:
        nodes, features = raw['node_ids'], raw['feature_names']
        if list(raw['data_model_ready'].shape) != meta['shape'] or meta['shape'][0] != meta['timeline_length']:
            raise ValueError('Raw dimensions differ from dataset manifest')
    dimensions = meta['shape'][1]*meta['shape'][2]
    for i, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as z:
            if json.loads(str(z['provenance_json'])) != config:
                raise ValueError(f'Graph provenance mismatch: {path}')
            if (int(z['graph_time_index']) != expected[i] or int(z['window_end_index']) != expected[i]
                    or int(z['window_start_index']) != starts[i]):
                raise ValueError(f'Graph time mismatch: {path}')
            if not (np.array_equal(z['node_ids'], nodes) and np.array_equal(z['feature_names'], features)):
                raise ValueError(f'Graph variable order mismatch: {path}')
            shape = (dimensions, dimensions, config['tau_max']+1)
            for key in ('p_matrix','q_matrix','val_matrix'):
                a = z[key]
                if a.shape != shape or not np.isfinite(a[:,:,1:]).all():
                    raise ValueError(f'Invalid complete lag tensor {key}: {path}')
                if key != 'val_matrix' and np.any((a[:,:,1:] < 0) | (a[:,:,1:] > 1)):
                    raise ValueError(f'Invalid probability range: {path}')
    return meta
