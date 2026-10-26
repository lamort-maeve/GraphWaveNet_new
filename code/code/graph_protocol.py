"""One graph scheduling contract shared by preparation, discovery and loading.

Pure NumPy module: discovery does not need PyTorch. All end indices are exclusive;
availability is an index at which observations strictly before it are available.
"""
import numpy as np


def graph_schedule(meta):
    if meta.get('schema') != 2:
        raise ValueError('Expected dataset schema 2; regenerate in a new directory with prepare_pilot.py')
    protocol = meta.get('graph_protocol')
    if protocol not in ('frozen_train', 'online'):
        raise ValueError('Graph protocol must be frozen_train or online')
    window, stride, total = (int(meta[k]) for k in ('graph_window','graph_stride','timeline_length'))
    if min(window, stride, total) < 1:
        raise ValueError('Invalid graph schedule dimensions')
    train = meta['splits']['train']
    ceiling = int(train['last_origin']) if protocol == 'frozen_train' else total-1
    cutoff = int(meta['discovery_cutoff'])
    if not window <= cutoff <= ceiling:
        raise ValueError('Discovery cutoff exceeds the declared protocol or precedes warmup')
    times = np.arange(window, cutoff+1, stride, dtype=np.int64)
    static = int(meta['static_available_index'])
    if static not in times or static >= int(train['end_exclusive']):
        raise ValueError('Static prior must be a scheduled graph strictly within training')
    return times-window, times.copy(), times
