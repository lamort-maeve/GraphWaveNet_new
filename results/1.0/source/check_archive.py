"""Forward every archived graph through the default eight-layer network on CPU."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from causal_model import DynamicGraphWaveNet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('prepared_run', type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    with np.load(args.prepared_run/'graphs/causal_tensor.npz') as z:
        bank = z['normalized']
    bank = np.concatenate([np.zeros_like(bank[:1]), bank])
    variables = bank.shape[-1]
    model = DynamicGraphWaveNet(variables, 1, list(range(variables)), bank,
                                hidden=4, blocks=4, layers=2, dropout=0).eval()
    visited = []
    with torch.no_grad():
        for start in range(1, len(bank), 8):
            indices = torch.arange(start, min(start+8, len(bank)))
            ids = indices[:, None].expand(-1, bank.shape[1])
            x = torch.randn(len(indices), 1, variables, bank.shape[1])
            y = model(x, ids)
            assert y.shape == (len(indices), 1, variables)
            assert torch.isfinite(y).all()
            visited.extend(indices.tolist())
    assert visited == list(range(1, len(bank)))
    # Full-width, full-depth backward check on one real causal graph.
    model = DynamicGraphWaveNet(variables, 1, list(range(variables)), bank,
                                hidden=32, blocks=4, layers=2, dropout=0)
    x = torch.randn(1, 1, variables, bank.shape[1], requires_grad=True)
    ids = torch.ones(1, bank.shape[1], dtype=torch.long)
    model(x, ids).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    report = dict(graphs_forward_checked=len(visited), graph_bank_ids=visited,
                  layers=8, variables=variables, lags=bank.shape[1],
                  full_width_backward_pass=True, device='cpu', full_training=False)
    (args.prepared_run/'all_graphs_forward_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'graph_bank_ids'}))


if __name__ == '__main__':
    main()
