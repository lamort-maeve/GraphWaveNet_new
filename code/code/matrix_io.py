"""Explicit, reusable static adjacency loading; never infer lag from a 2-D graph."""
import pickle
from pathlib import Path

import numpy as np


def load_matrix(path, node_ids, features, variables, *, trusted_pickle=False,
                assume_order=False, key='adjacency'):
    """Read NPY/NPZ/CSV or GraphWaveNet (ids, index, adjacency) pickle.

    Axes are source,target. Road matrices are placed on flow-to-flow edges
    when the model uses all variables. No edges for other features are invented.
    Variable matrices use explicit `road:feature` IDs, or an order declaration.
    """
    path = Path(path)
    ids = None
    if path.suffix.lower() in ('.pkl', '.pickle'):
        if not trusted_pickle:
            raise ValueError('Pickle matrices require --trusted-matrix-pickle')
        with path.open('rb') as stream:
            value = pickle.load(stream, encoding='latin1')
        if isinstance(value, tuple) and len(value) == 3:
            ids, _, value = value
        elif isinstance(value, dict):
            ids, value = value.get('node_ids'), value[key]
        matrix = np.asarray(value, dtype=np.float32)
    elif path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            matrix = np.asarray(archive[key], dtype=np.float32)
            ids = archive['node_ids'] if 'node_ids' in archive else None
    elif path.suffix.lower() == '.npy':
        matrix = np.load(path, allow_pickle=False).astype(np.float32)
    elif path.suffix.lower() == '.csv':
        matrix = np.loadtxt(path, delimiter=',', dtype=np.float32)
    else:
        raise ValueError('Matrix format must be .npy, .npz, .csv, .pkl or .pickle')
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError('Expected a finite square source,target matrix')
    roads = [str(v) for v in node_ids]
    variable_ids = [f'{road}:{feature}' for road in roads for feature in features]
    if matrix.shape[0] == len(roads):
        expected = roads
    elif matrix.shape[0] == len(variable_ids):
        expected = variable_ids
    else:
        raise ValueError(f'Matrix shape {matrix.shape} does not match roads or variables')
    if ids is None:
        if not assume_order:
            raise ValueError('Matrix has no IDs; supply --matrix-assume-order after checking source,target order')
    else:
        ids = [v.decode() if isinstance(v, bytes) else str(v) for v in ids]
        if len(set(ids)) != len(ids) or len(ids) != len(expected) or set(ids) != set(expected):
            raise ValueError('Matrix node IDs differ from the dataset IDs')
        order = [ids.index(v) for v in expected]
        matrix = matrix[np.ix_(order, order)]
    flow = list(features).index('flow')
    flow_indices = np.arange(len(roads)) * len(features) + flow
    if len(expected) == len(roads) and variables == 'all':
        lifted = np.zeros((len(variable_ids), len(variable_ids)), dtype=np.float32)
        lifted[np.ix_(flow_indices, flow_indices)] = matrix
        matrix = lifted
    elif len(expected) == len(variable_ids) and variables == 'flow':
        matrix = matrix[np.ix_(flow_indices, flow_indices)]
    return np.ascontiguousarray(matrix)


def reference_supports(matrix, adjtype):
    """Original GraphWaveNet row-normalized diffusion supports (no PCMCI processing)."""
    if adjtype == 'raw':
        return [matrix]
    if adjtype == 'identity':
        return [np.eye(len(matrix), dtype=np.float32)]
    if np.any(matrix < 0):
        raise ValueError('Transition supports require nonnegative weights; use raw for signed matrices')
    def transition(a):
        degree = a.sum(1, keepdims=True)
        return np.divide(a, degree, out=np.zeros_like(a), where=degree != 0)
    if adjtype == 'transition':
        return [transition(matrix)]
    if adjtype == 'doubletransition':
        return [transition(matrix), transition(matrix.T)]
    raise ValueError(adjtype)
