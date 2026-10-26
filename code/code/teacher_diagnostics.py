"""Exploratory diagnostic on withheld training-origin targets only."""
import csv
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from data_pipeline import Samples
from graph_reliability import reliability_table, lag_reliability_table


@torch.no_grad()
def diagnose(teacher, data, archive, supervision, args, run, mean, std, diagnostic_archive=None):
    teacher.eval()
    probe_archive = diagnostic_archive if diagnostic_archive is not None else archive
    variation, _ = reliability_table(probe_archive.bank, args.reliability_history, args.reliability_gamma)
    definition = 'legacy_relative_signed_L1'
    if getattr(args, 'ssl_method', None) == 'graph_mean_teacher_v3':
        _, details = lag_reliability_table(probe_archive.bank, args.reliability_history,
            args.reliability_gamma, args.confidence_floor)
        variation = details['structural_score']
        definition = 'v3_lag_balanced_past_median_MAD_excess'
    split = 'diagnostic' if 'diagnostic' in data.splits else 'train'
    origins = data.splits[split]
    indices = np.arange(len(origins)) if split == 'diagnostic' else np.flatnonzero(~supervision[origins].numpy())
    if args.smoke:
        indices = indices[np.linspace(0, max(0, len(indices)-1), min(len(indices), args.batch_size*2), dtype=int)]
    loader = DataLoader(Subset(Samples(data, archive, split), indices.tolist()), batch_size=args.batch_size)
    rows = []
    for x, y, valid, ids, origin in loader:
        pred = (teacher(x.to(mean.device), ids.to(mean.device))*std + mean).cpu()
        error = torch.where(valid, (pred-y).abs(), torch.zeros_like(y)).sum((1, 2))
        count = valid.sum((1, 2))
        probe_ids = probe_archive.ids(origin.numpy())
        for t, g, teacher_g, e, n in zip(origin.tolist(), probe_ids.tolist(), ids[:, -1].tolist(), error.tolist(), count.tolist()):
            if n:
                rows.append(dict(origin=t, graph_id=g, teacher_graph_id=teacher_g, variation=float(variation[g]),
                                 absolute_error_sum=e, observed_count=n, teacher_mae=e/n))
    with (run/'teacher_diagnostic.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['origin', 'graph_id', 'teacher_graph_id', 'variation', 'absolute_error_sum', 'observed_count', 'teacher_mae'])
        writer.writeheader(); writer.writerows(rows)
    # One row per graph interval avoids treating a repeated stability score as
    # thousands of independent pieces of structural evidence. Still correlated.
    grouped = {}
    for row in rows:
        if row['graph_id'] < 2:
            continue
        sums = grouped.setdefault(row['graph_id'], [0., 0.])
        sums[0] += row['absolute_error_sum']; sums[1] += row['observed_count']
    x = np.array([variation[g] for g in grouped])
    y = np.array([v[0]/v[1] for v in grouped.values()])
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else None
    result = dict(split='purged_training_tail' if split == 'diagnostic' else 'withheld_training_origin_targets', rows=len(rows), graph_intervals=len(grouped),
                  probe='as-of graph drift in original training period, diagnostic only; teacher uses runtime graph policy',
                  correlation_identifiable=bool(corr is not None),
                  variation_definition=definition,
                  pearson_graph_interval=corr, selection='best student validation epoch; matching EMA teacher',
                  limitation=('Chronological diagnostic holdout; model scaler fit only before holdout. Raw imputation and PCMCI preprocessing still require upstream provenance auditing. No iid p-value.' if split == 'diagnostic' else
                              'Exploratory, not an independent holdout: overlapping historical observations remain visible. No iid p-value. Use diagnostic-fraction before claiming mechanism.'))
    (run/'teacher_diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
