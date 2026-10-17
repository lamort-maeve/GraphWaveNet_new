"""Complete PCMCI archive and chronological, leakage-safe sample alignment."""
import csv
import hashlib
import json
import os
import uuid
from zipfile import BadZipFile
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
    def __init__(self, directory, node_ids, features, alpha=0.05, variables='all', weighting='confidence'):
        if weighting not in ('confidence', 'legacy'):
            raise ValueError(weighting)
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
                # q is statistical evidence, NOT a probability of causality.
                # Retain significant edges but discount those near the threshold.
                reliability = np.where(np.isfinite(q) & (q >= 0) & (q <= alpha),
                                       1. - 0.5 * np.clip(q / alpha, 0., 1.), 0.)
                weighted = a * reliability[selected][:, selected].transpose(2, 0, 1)
                entries.append((available, a, meta, weighted.astype(np.float32)))
        entries.sort(key=lambda x: x[0])
        self.times = np.array([e[0] for e in entries], dtype=np.int64)
        if np.any(np.diff(self.times) <= 0):
            raise ValueError('Graph availability times must be unique')
        self.raw = np.stack([e[1] for e in entries])
        # Incoming absolute-degree normalization for each lag; zero degree stays zero.
        values = self.raw if weighting == 'legacy' else np.stack([e[3] for e in entries])
        # A weak singleton edge must not become a unit edge. Confidence mode
        # caps incoming mass at one without amplifying weak neighborhoods.
        floor = 1e-12 if weighting == 'legacy' else 1.
        norm = values / np.maximum(np.abs(values).sum(axis=2, keepdims=True), floor)
        # Index zero means no causal graph has become available yet.
        self.bank = np.concatenate([np.zeros_like(norm[:1]), norm], axis=0)
        slots = [e[2]['slot'] for e in entries]
        self.metadata = dict(graph_count=len(entries), shape=list(self.raw.shape),
                             axes=['window', 'lag', 'source_variable', 'target_variable'],
                             variables=variables, node_ids=list(map(str, node_ids)),
                             features=list(map(str, features)), alpha=alpha, weighting=weighting,
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
    def __init__(self, data_dir, raw_path, variables='all', context=25, trusted_object_metadata=False):
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
        self.data, self.split = data, split
        # Workers need graph IDs, not the entire raw + normalized graph archive.
        self.timeline_graph_ids = archive.ids(np.arange(len(data.values)))

    def __len__(self):
        return len(self.data.splits[self.split])

    def __getitem__(self, index):
        d = self.data
        origin = d.splits[self.split][index]
        start = int(origin) - d.context + 1
        if start >= 0:
            x = d.values[start:origin+1]
            ids = self.timeline_graph_ids[start:origin+1]
        else:
            x = np.zeros((d.context, d.values.shape[1]), dtype=d.values.dtype)
            ids = np.zeros(d.context, dtype=np.int64)
            x[-start:] = d.values[:origin+1]
            ids[-start:] = self.timeline_graph_ids[:origin+1]
        return (torch.from_numpy(x.T[None]),
                torch.from_numpy(d.targets[self.split][index]),
                torch.from_numpy(d.masks[self.split][index]),
                torch.from_numpy(ids), int(origin))


def pack_cache(state):
    """Array + JSON archive only; never pickle objects from a cache."""
    arrays = {}
    def encode(value):
        if isinstance(value, np.ndarray):
            key = 'array_' + str(len(arrays))
            arrays[key] = value
            return {'__array__': key}
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value
    arrays['metadata'] = np.asarray(json.dumps(encode(state), ensure_ascii=False))
    return arrays


def unpack_cache(archive):
    def decode(value):
        if isinstance(value, dict):
            if set(value) == {'__array__'}:
                return archive[value['__array__']]
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value
    return decode(json.loads(str(archive['metadata'])))


def load_prepared(args):
    """Reuse validated preprocessing while files and preprocessing settings match."""
    needs_graphs = not (args.architecture in ('gwnet', 'paper') and args.mode == 'adaptive')
    graph_paths = sorted(args.graphs.glob('graph_*.npz')) if needs_graphs else []
    paths = [args.raw] + [args.data/(s+'.npz') for s in ('train', 'val', 'test')]
    paths += graph_paths + [p.with_name(p.stem+'_edges.csv') for p in graph_paths]
    fingerprint = dict(version=2, source=digest(Path(__file__)), needs_graphs=needs_graphs,
                       options=[args.variables, args.context, args.alpha, args.graph_weighting,
                                args.trusted_object_metadata],
                       files=[(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths])
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    cache_dir = args.cache_dir or args.output/'cache'
    cache_path = cache_dir/(key+'.npz')
    if not args.no_data_cache and not args.rebuild_cache and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as z:
                state = unpack_cache(z)
            graph = GraphArchive.__new__(GraphArchive)
            data = TrafficData.__new__(TrafficData)
            graph.__dict__.update(state['graph'])
            data.__dict__.update(state['data'])
            print(f'Prepared data cache HIT: {cache_path}', flush=True)
            return data, graph, state['inputs']
        except (ValueError, KeyError, OSError, EOFError, BadZipFile):
            print('Prepared cache invalid; rebuilding from original inputs.', flush=True)
    if needs_graphs:
        with np.load(args.raw, allow_pickle=args.trusted_object_metadata) as z:
            graph = GraphArchive(args.graphs, z['node_ids'], z['feature_names'], args.alpha,
                                 args.variables, weighting=args.graph_weighting)
    else:
        # Pure reference adaptive training never opens or validates PCMCI files.
        graph = GraphArchive.__new__(GraphArchive)
        graph.times = np.array([], dtype=np.int64)
        graph.bank = np.zeros((1, 1, 1, 1), dtype=np.float32)
        graph.raw = graph.bank[:0]
        graph.metadata = dict(graph_count=0, missing_slots=[], windows=[], weighting='unused')
    context = args.context if args.architecture == 'paper' else max(args.context, graph.bank.shape[1])
    data = TrafficData(args.data, args.raw, args.variables, context,
                       args.trusted_object_metadata)
    inputs = dict(raw=dict(path=str(args.raw.resolve()), sha256=digest(args.raw)),
                  splits={s: dict(path=str((args.data/(s+'.npz')).resolve()), sha256=digest(args.data/(s+'.npz')))
                          for s in ('train', 'val', 'test')},
                  graph_files={r['file']: r['sha256'] for r in graph.metadata['windows']})
    if not args.no_data_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_dir/(key+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with open(temporary, 'wb') as f:
                np.savez(f, **pack_cache(dict(graph=graph.__dict__, data=data.__dict__, inputs=inputs)))
            os.replace(temporary, cache_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f'Prepared data cache CREATED: {cache_path}', flush=True)
    return data, graph, inputs
