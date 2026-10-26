"""Masked forecast metrics and loss; observed zero targets remain valid."""
import torch

class Metrics:
    def __init__(self, horizon):
        self.sums = torch.zeros((horizon, 5), dtype=torch.float64)

    @torch.no_grad()
    def update(self, pred, truth, valid):
        # True zero flow is a valid target; only MAPE excludes zero denominators.
        diff = torch.where(valid, pred-truth, torch.zeros_like(pred)).double()
        pct = valid & (truth.abs() > 1e-5)
        vals = [diff.abs().sum((0, 2)), diff.square().sum((0, 2)), valid.sum((0, 2)),
                torch.where(pct, diff.abs()/truth.abs().clamp_min(1e-5), torch.zeros_like(diff)).sum((0, 2)),
                pct.sum((0, 2))]
        # Accumulate on device; one CPU transfer per epoch, not per batch.
        self.sums = self.sums.to(diff.device)
        self.sums += torch.stack(vals, -1).detach()

    @staticmethod
    def row(s):
        a, q, count, pct, pct_count = map(float, s)
        return dict(mae=a/count if count else None, rmse=(q/count)**0.5 if count else None,
                    mape=pct/pct_count if pct_count else None, valid_count=int(count),
                    mape_count=int(pct_count))

    def result(self):
        sums = self.sums.cpu().numpy()
        return dict(overall=self.row(sums.sum(0)),
                    by_horizon=[dict(horizon=i+1, **self.row(s)) for i, s in enumerate(sums)])

def masked_mae(pred, truth, valid):
    # Boolean indexing creates a dynamic-sized CUDA output (and a host sync).
    # A fixed-shape reduction has the same objective, including true zero targets.
    return torch.where(valid, (pred-truth).abs(), torch.zeros_like(pred)).sum() / valid.sum().clamp_min(1)
