"""Complete PCMCI archive and chronological, leakage-safe sample alignment."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def decode_strings(values):
    return np.asarray([v.decode('utf-8') if isinstance(v, (bytes, np.bytes_)) else str(v) for v in values])


class GraphArchive:
    def __init__(self, directory, node_ids, features, alpha=0.05, variables='all'):
        node_ids, features = decode_strings(node_ids), decode_strings(features)
        if not 0 < alpha <= 1:
            raise ValueError('alpha must be in (0, 1]')
        paths = list(Path(directory).glob('graph_*.npz'))
        if not paths:
            raise ValueError(f'No graph files in {directory}')
        feature_index = {str(v): i for i, v in enumerate(features)}
        n, f = len(node_ids), len(features)
        selected = np.arange(n * f) if variables == 'all' else np.arange(n) * f + feature_index['flow']
        entries = []
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                v, q = z['val_matrix'], z['q_matrix']
                if v.shape != q.shape or v.shape[:2] != (n*f, n*f):
                    raise ValueError(f'Invalid PCMCI shape: {path}')
                start, end, available = (int(z[k]) for k in
                                        ('window_start_index', 'window_end_index', 'graph_time_index'))
                if not 0 <= start < end <= available:
                    raise ValueError(f'Invalid availability times: {path}')
                # Validate variable order and direction against the exported edge list.
                edge_path = path.with_name(path.stem + '_edges.csv')
                if not edge_path.exists():
                    raise ValueError(f'Missing variable-order evidence: {edge_path}')
                with open(edge_path, encoding='utf-8-sig', newline='') as stream:
                    for row in csv.DictReader(stream):
                        si, ti = int(row['source_node_index']), int(row['target_node_index'])
                        if str(node_ids[si]) != row['source_node_id'] or str(node_ids[ti]) != row['target_node_id']:
                            raise ValueError(f'Node ID mismatch: {edge_path}')
                        i = si*f + feature_index[row['source_feature']]
                        j = ti*f + feature_index[row['target_feature']]
                        lag = int(row['lag_5min_steps'])
                        if not np.isclose(v[i, j, lag], float(row['parcorr_value']), atol=1e-6):
                            raise ValueError(f'Variable order / edge weight mismatch: {edge_path}')
                # No top-k truncation or window averaging. Signed effects and all lags survive.
                a = np.where(np.isfinite(q) & (q <= alpha) & np.isfinite(v), v, 0)
                a = a[selected][:, selected].transpose(2, 0, 1).astype(np.float32)
                meta = dict(file=path.name, slot=int(path.stem.split('_')[1]),
                            start=start, end=end, available=available,
                            sha256=digest(path), edges_sha256=digest(edge_path),
                            significant_entries=int(np.count_nonzero(a)))
                entries.append((available, a, meta))
        entries.sort(key=lambda x: x[0])
        self.times = np.array([e[0] for e in entries], dtype=np.int64)
        if np.any(np.diff(self.times) <= 0):
            raise ValueError('Graph availability times must be unique')
        self.raw = np.stack([e[1] for e in entries])
        # Incoming absolute-degree normalization for each lag; zero degree stays zero.
        norm = self.raw / np.maximum(np.abs(self.raw).sum(axis=2, keepdims=True), 1e-12)
        # Index zero means no causal graph has become available yet.
        self.bank = np.concatenate([np.zeros_like(norm[:1]), norm], axis=0)
        slots = [e[2]['slot'] for e in entries]
        self.metadata = dict(graph_count=len(entries), shape=list(self.raw.shape),
                             axes=['window', 'lag', 'source_variable', 'target_variable'],
                             variables=variables, node_ids=list(map(str, node_ids)),
                             features=list(map(str, features)), alpha=alpha,
                             missing_slots=sorted(set(range(max(slots)+1))-set(slots)),
                             windows=[e[2] for e in entries])

    def ids(self, time_indices):
        # As-of lookup: missing windows carry the last *past* graph, never a future graph.
        ids = np.searchsorted(self.times, time_indices, side='right')
        ids = np.where(time_indices < 0, 0, ids)
        return ids.astype(np.int64)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / 'causal_tensor.npz', signed=self.raw,
                            normalized=self.bank[1:], available=self.times)
        (directory / 'manifest.json').write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')


class TrafficData:
    def __init__(self, data_dir, raw_path, variables='all', context=25, trusted_object_metadata=True):
        with np.load(raw_path, allow_pickle=trusted_object_metadata) as z:
            raw = z['data_model_ready'].astype(np.float32)
            self.node_ids, self.features = z['node_ids'], z['feature_names']
            observed = z['target_observed_mask'].astype(bool)
        self.node_ids, self.features = decode_strings(self.node_ids), decode_strings(self.features)
        flow = list(self.features).index('flow')
        self.flow_indices = np.arange(raw.shape[1]) * raw.shape[2] + flow
        self.splits, self.targets, self.masks = {}, {}, {}
        cursor, first_offsets, first_y = 0, None, None
        for name in ('train', 'val', 'test'):
            with np.load(Path(data_dir) / (name + '.npz'), allow_pickle=False) as z:
                x, y = z['x'], z['y']
                xo, yo = z['x_offsets'].ravel(), z['y_offsets'].ravel()
            if not np.array_equal(xo, np.arange(-len(xo)+1, 1)) or np.any(yo <= 0):
                raise ValueError('Expected contiguous historical inputs and positive target offsets')
            if first_offsets is None:
                first_offsets, first_y = xo, yo
            if not np.array_equal(first_offsets, xo) or not np.array_equal(first_y, yo):
                raise ValueError('Inconsistent offsets between splits')
            origins = np.arange(cursor, cursor + len(x)) - int(xo[0])
            if origins[-1] + max(yo) >= len(raw):
                raise ValueError('Dataset exceeds raw timeline')
            # Exhaustive, chunked alignment validation; fail rather than assume shuffled data is chronological.
            for lo in range(0, len(x), 2048):
                ts = origins[lo:lo+2048]
                if not np.allclose(x[lo:lo+2048, ..., 0], raw[ts[:, None]+xo, :, flow], atol=1e-5):
                    raise ValueError(f'{name}: x does not align with raw data')
                if not np.allclose(y[lo:lo+2048, ..., 0], raw[ts[:, None]+yo, :, flow], atol=1e-5):
                    raise ValueError(f'{name}: y does not align with raw data')
            self.splits[name] = origins
            self.targets[name] = y[..., 0].astype(np.float32)
            self.masks[name] = observed[origins[:, None]+yo] & np.isfinite(y[..., 0])
            cursor += len(x)
        if cursor != len(raw) + int(first_offsets[0]) - int(max(first_y)):
            raise ValueError('Split lengths do not cover the raw timeline')
        self.context = max(context, len(first_offsets))
        self.horizon = len(first_y)
        self.y_offsets = first_y
        self.train_cutoff = int(self.splits['train'][-1])
        # Purge training targets overlapping validation inputs; likewise validation/test.
        # All modes use these exact same examples.
        for left, right in (('train', 'val'), ('val', 'test')):
            cutoff = max(0, int(self.splits[right][0]) - self.context + 1)
            keep = self.splits[left] + max(first_y) < cutoff
            self.splits[left] = self.splits[left][keep]
            self.targets[left] = self.targets[left][keep]
            self.masks[left] = self.masks[left][keep]
        train_end = int(self.splits['train'][-1]) + 1
        selected = raw.reshape(len(raw), -1)
        if variables == 'flow':
            selected = selected[:, self.flow_indices]
            self.flow_indices = np.arange(raw.shape[1])
        if not np.isfinite(selected).all():
            raise ValueError('Nonfinite model inputs')
        self.mean = selected[:train_end].mean(axis=0)
        self.std = np.maximum(selected[:train_end].std(axis=0), 1e-5)
        self.values = (selected - self.mean) / self.std
        self.output_mean = self.mean[self.flow_indices]
        self.output_std = self.std[self.flow_indices]


class Samples(Dataset):
    def __init__(self, data, archive, split):
        self.data, self.archive, self.split = data, archive, split

    def __len__(self):
        return len(self.data.splits[self.split])

    def __getitem__(self, index):
        d = self.data
        origin = d.splits[self.split][index]
        ts = origin + np.arange(-d.context+1, 1)
        x = d.values[np.maximum(ts, 0)].copy()
        x[ts < 0] = 0
        return (torch.from_numpy(x.T[None]),
                torch.from_numpy(d.targets[self.split][index]),
                torch.from_numpy(d.masks[self.split][index]),
                torch.from_numpy(self.archive.ids(ts)), int(origin))
