"""Optional execution optimizations; no change to labels, ordering or precision."""
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class BatchSamples(Dataset):
    """Fetch one list of sample indices with NumPy instead of a Python sample loop.

    Use with batch_size=None and a BatchSampler passed as sampler. The same
    sampler generator and DataLoader base-seed consumption preserve ordering.
    Integer indexing remains available for inspection and horizon inference.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (int, np.integer)):
            return self.dataset[int(indices)]
        dataset = self.dataset
        indices = np.asarray(indices, dtype=np.int64)
        while isinstance(dataset, Subset):
            indices = np.asarray(dataset.indices)[indices]
            dataset = dataset.dataset
        data, split = dataset.data, dataset.split
        origins = data.splits[split][indices]
        times = origins[:, None] + np.arange(-data.context+1, 1)
        available = times >= 0
        safe = times.clip(min=0)
        values = data.values[safe].copy()
        values[~available] = 0
        ids = dataset.timeline_graph_ids[safe].copy()
        ids[~available] = 0
        return (torch.from_numpy(np.ascontiguousarray(values.transpose(0, 2, 1)[:, None])),
                torch.from_numpy(np.ascontiguousarray(data.targets[split][indices])),
                torch.from_numpy(np.ascontiguousarray(data.masks[split][indices])),
                torch.from_numpy(np.ascontiguousarray(ids)),
                torch.from_numpy(np.ascontiguousarray(origins, dtype=np.int64)))


@torch.no_grad()
def uniform_history_bank(bank, kernel, padding):
    """Cache fixed signed uniform averages, with the original K reduction order.

    Every row is made only from that row and earlier IDs. Lag propagation and
    trainable lag weights are NOT cached or reordered. Work is once per bank
    version, outside the per-layer/per-minibatch hot path.
    """
    offsets = torch.arange(kernel-1, -1, -1, device=bank.device)
    prior = (torch.arange(len(bank), device=bank.device)[:, None]-offsets).clamp_min(0)
    weights = bank.new_zeros(kernel).softmax(0)[None].expand(len(bank), -1)
    if padding == 'masked':
        weights = weights*(prior > 0)
        weights = weights/weights.sum(1, keepdim=True).clamp_min(1e-12)
    positive, negative = [], []
    for lag in range(bank.shape[1]):
        edge = bank[:, lag][prior]
        positive.append((edge.clamp_min(0)*weights[:, :, None, None]).sum(1))
        negative.append(((-edge).clamp_min(0)*weights[:, :, None, None]).sum(1))
    return torch.stack(positive, 1), torch.stack(negative, 1)


class FixedGraphCache:
    """Ephemeral cache shared by student/teacher, excluded from checkpoints.

    Version + identity invalidate after load_state_dict, copy_, zero_ or .to().
    Do not mutate tensors through .data: that bypasses PyTorch version tracking.
    """
    def __init__(self):
        self.key = None
        self.signed = None
        self.uniform = None

    def get(self, bank, kernel, padding, need_uniform=False):
        key = (id(bank), bank._version, bank.device, bank.dtype, tuple(bank.shape),
               kernel, padding, need_uniform)
        if self.key != key:
            with torch.no_grad():
                self.signed = (bank.clamp_min(0), (-bank).clamp_min(0))
                self.uniform = uniform_history_bank(bank, kernel, padding) if need_uniform else None
            self.key = key
        return self.signed, self.uniform
