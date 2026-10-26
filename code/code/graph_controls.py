"""Graph ablations preserve graph availability and never mutate the cached archive."""
import copy
import numpy as np

def graph_control(archive, name, seed):
    """Deterministic negative controls; leave the cached archive untouched."""
    if name in ('dynamic', 'hybrid') or not len(archive.times):
        return archive
    result = copy.copy(archive)
    bank = archive.bank.copy()
    if name == 'static':
        # Warm-up graph only, available at the same first timestamp. A full
        # training-period average would read future windows for early samples.
        bank[1:] = bank[1]
    elif name == 'shuffled':
        order = np.random.default_rng(seed).permutation(bank.shape[-2])
        bank = bank[:, :, order, :]
    elif name == 'collapsed':
        # Deliberately discard delay identity for the lag-preservation ablation.
        collapsed = bank[:, 1:].sum(1)
        bank.fill(0)
        if bank.shape[1] > 1:
            bank[:, 1] = collapsed / np.maximum(np.abs(collapsed).sum(-2, keepdims=True), 1.)
    elif name == 'local':
        bank.fill(0)
        bank[1:, 1:] = np.eye(bank.shape[-1], dtype=np.float32)
    elif name == 'zero':
        bank.fill(0)
    elif name == 'unsigned':
        bank = np.abs(bank)
    else:
        raise ValueError(f'Unknown graph control: {name}')
    result.bank = bank
    return result
