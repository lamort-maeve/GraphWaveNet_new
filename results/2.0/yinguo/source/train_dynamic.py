# """Server training entry point. --prepare-only never trains; --smoke is bounded."""

# import argparse
# import csv
# import json
# import random
# import shutil
# import time
# import zipfile
# from pathlib import Path

# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Subset

# from causal_data import GraphArchive, TrafficData, Samples, digest
# from causal_model import DynamicGraphWaveNet


# ROOT = Path(__file__).resolve().parents[2]


# def parser():
#     p = argparse.ArgumentParser(description=__doc__)

#     p.add_argument("--data", type=Path, default=ROOT / "data" / "data1")
#     p.add_argument(
#         "--raw",
#         type=Path,
#         default=ROOT / "data" / "data1" / "full_year_cleaned.npz",
#     )
#     p.add_argument("--graphs", type=Path, default=ROOT / "data" / "per_window")
#     p.add_argument("--output", type=Path, default=ROOT / "result" / "result1")

#     p.add_argument(
#         "--mode",
#         choices=["causal", "adaptive", "combined", "all"],
#         default="all",
#     )
#     p.add_argument("--variables", choices=["all", "flow"], default="all")
#     p.add_argument("--device", default="cuda")
#     p.add_argument(
#         "--gpus",
#         type=int,
#         default=4,
#         help="Number of GPUs to use with DataParallel; ignored on CPU",
#     )
#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--epochs", type=int, default=100)
#     p.add_argument("--patience", type=int, default=20)
#     p.add_argument("--batch-size", type=int, default=32)
#     p.add_argument("--hidden", type=int, default=32)
#     p.add_argument("--blocks", type=int, default=4)
#     p.add_argument("--layers", type=int, default=2)
#     p.add_argument("--context", type=int, default=13)
#     p.add_argument("--learning-rate", type=float, default=0.001)
#     p.add_argument("--weight-decay", type=float, default=0.0001)
#     p.add_argument("--dropout", type=float, default=0.3)
#     p.add_argument("--alpha", type=float, default=0.05)
#     p.add_argument("--threads", type=int, default=2)
#     p.add_argument("--workers", type=int, default=0)
#     p.add_argument("--prepare-only", action="store_true")
#     p.add_argument(
#         "--smoke",
#         action="store_true",
#         help="Only 2 train batches and 2 eval batches per mode",
#     )
#     p.add_argument("--run-name", default=None)
#     return p


# def seed_all(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True


# def write_json(path, value):
#     Path(path).write_text(
#         json.dumps(value, indent=2, ensure_ascii=False, default=str),
#         encoding="utf-8",
#     )


# def resolve_device(args):
#     """Return the primary device and wrap-ready device list."""
#     if args.device.startswith("cuda"):
#         if not torch.cuda.is_available():
#             raise RuntimeError("--device cuda requested but CUDA is unavailable")

#         count = torch.cuda.device_count()
#         if count < 1:
#             raise RuntimeError("No CUDA devices found")

#         use = min(args.gpus, count)
#         if use < 1:
#             raise ValueError("--gpus must be >= 1 when using CUDA")

#         print(
#             f"CUDA devices available={count}; using {use} GPU(s)",
#             flush=True,
#         )
#         return torch.device("cuda:0"), use

#     return torch.device(args.device), 0


# def wrap_model(model, device, gpu_count):
#     """Wrap with DataParallel when more than one GPU is used."""
#     if gpu_count > 1:
#         model = torch.nn.DataParallel(
#             model,
#             device_ids=list(range(gpu_count)),
#             output_device=0,
#         )

#     return model.to(device)


# def unwrap_model(model):
#     return model.module if isinstance(model, torch.nn.DataParallel) else model


# class Metrics:
#     def __init__(self, horizon):
#         self.sums = np.zeros((horizon, 5), np.float64)

#     def update(self, pred, truth, valid):
#         # 零流量也是有效目标；只有 MAPE 排除零分母。
#         diff = torch.where(
#             valid, pred - truth, torch.zeros_like(pred)
#         ).double()
#         pct = valid & (truth.abs() > 1e-5)

#         vals = [
#             diff.abs().sum((0, 2)),
#             diff.square().sum((0, 2)),
#             valid.sum((0, 2)),
#             torch.where(
#                 pct,
#                 diff.abs() / truth.abs().clamp_min(1e-5),
#                 torch.zeros_like(diff),
#             ).sum((0, 2)),
#             pct.sum((0, 2)),
#         ]

#         self.sums += torch.stack(vals, -1).detach().cpu().numpy()

#     @staticmethod
#     def row(s):
#         a, q, count, pct, pct_count = map(float, s)
#         return {
#             "mae": a / count if count else None,
#             "rmse": (q / count) ** 0.5 if count else None,
#             "mape": pct / pct_count if pct_count else None,
#             "valid_count": int(count),
#             "mape_count": int(pct_count),
#         }

#     def result(self):
#         return {
#             "overall": self.row(self.sums.sum(0)),
#             "by_horizon": [
#                 dict(horizon=i + 1, **self.row(s))
#                 for i, s in enumerate(self.sums)
#             ],
#         }


# def pass_epoch(
#     model,
#     loader,
#     device,
#     mean,
#     std,
#     optimizer=None,
#     capture=False,
# ):
#     training = optimizer is not None
#     model.train(training)

#     metric = Metrics(len(loader.dataset[0][1]))
#     predictions, targets, masks, origins = [], [], [], []

#     with torch.set_grad_enabled(training):
#         for x, y, valid, graph_ids, origin in loader:
#             x, y, valid, graph_ids = (
#                 v.to(device) for v in (x, y, valid, graph_ids)
#             )

#             pred = model(x, graph_ids) * std + mean

#             if not torch.isfinite(pred).all():
#                 raise FloatingPointError("Nonfinite predictions")

#             if training:
#                 if not valid.any():
#                     continue

#                 loss = (pred - y).abs()[valid].mean()
#                 optimizer.zero_grad(set_to_none=True)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(
#                     model.parameters(),
#                     5.0,
#                     error_if_nonfinite=True,
#                 )
#                 optimizer.step()

#             metric.update(pred, y, valid)

#             if capture:
#                 predictions.append(pred.detach().cpu().numpy())
#                 targets.append(y.cpu().numpy())
#                 masks.append(valid.cpu().numpy())
#                 origins.append(origin.numpy())

#     packed = None
#     if capture:
#         packed = {
#             "prediction": np.concatenate(predictions),
#             "target": np.concatenate(targets),
#             "observed": np.concatenate(masks),
#             "origin_index": np.concatenate(origins),
#         }

#     return metric.result(), packed


# def make_loader(data, archive, split, args):
#     dataset = Samples(data, archive, split)

#     if args.smoke:
#         # 跨时间线取样，覆盖后期图，而不只是没有图的初始阶段。
#         indices = np.linspace(
#             0,
#             len(dataset) - 1,
#             min(len(dataset), args.batch_size * 2),
#             dtype=int,
#         )
#         dataset = Subset(dataset, indices.tolist())

#     return DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=split == "train",
#         num_workers=args.workers,
#         drop_last=False,
#         generator=torch.Generator().manual_seed(args.seed),
#     )


# def run_mode(args, data, archive, directory, mode):
#     seed_all(args.seed)

#     run = directory / mode
#     run.mkdir(parents=True, exist_ok=False)

#     device, gpu_count = resolve_device(args)
#     model = DynamicGraphWaveNet(
#         data.values.shape[1],
#         data.horizon,
#         data.flow_indices,
#         archive.bank,
#         mode,
#         args.hidden,
#         args.blocks,
#         args.layers,
#         args.dropout,
#     )
#     model = wrap_model(model, device, gpu_count)

#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=args.learning_rate,
#         weight_decay=args.weight_decay,
#     )

#     loaders = {
#         split: make_loader(data, archive, split, args)
#         for split in ("train", "val", "test")
#     }

#     mean = torch.as_tensor(data.output_mean, device=device)
#     std = torch.as_tensor(data.output_std, device=device)

#     config = dict(
#         vars(args),
#         mode=mode,
#         context=data.context,
#         horizon=data.horizon,
#         variables_count=data.values.shape[1],
#         parameters=sum(p.numel() for p in model.parameters()),
#         torch_version=str(torch.__version__),
#         numpy_version=np.__version__,
#         gpu_count=gpu_count,
#     )
#     config = json.loads(json.dumps(config, default=str))
#     write_json(run / "config.json", config)

#     best, stale, history = float("inf"), 0, []
#     epochs = 1 if args.smoke else args.epochs

#     for epoch in range(1, epochs + 1):
#         start = time.time()

#         train, _ = pass_epoch(
#             model, loaders["train"], device, mean, std, optimizer
#         )
#         val, _ = pass_epoch(
#             model, loaders["val"], device, mean, std
#         )

#         score = val["overall"]["mae"]
#         if score is None or not np.isfinite(score):
#             raise ValueError("Validation has no finite observed-target metric")

#         row = dict(
#             epoch=epoch,
#             seconds=time.time() - start,
#             **{"train_" + k: v for k, v in train["overall"].items()},
#             **{"val_" + k: v for k, v in val["overall"].items()},
#         )
#         history.append(row)

#         with open(
#             run / "history.csv", "w", newline="", encoding="utf-8"
#         ) as f:
#             writer = csv.DictWriter(f, fieldnames=list(row))
#             writer.writeheader()
#             writer.writerows(history)

#         print(
#             f'{mode} epoch={epoch} '
#             f'train_MAE={train["overall"]["mae"]:.4f} '
#             f'val_MAE={score:.4f}',
#             flush=True,
#         )

#         if score < best:
#             best, stale = score, 0
#             torch.save(
#                 {
#                     "model": unwrap_model(model).state_dict(),
#                     "optimizer": optimizer.state_dict(),
#                     "epoch": epoch,
#                     "validation": val,
#                     "config": config,
#                     "scaler_mean": torch.from_numpy(data.mean),
#                     "scaler_std": torch.from_numpy(data.std),
#                     "graph_times": torch.from_numpy(archive.times),
#                 },
#                 run / "best.pt",
#             )
#         else:
#             stale += 1

#         if args.patience and stale >= args.patience:
#             break

#     checkpoint = torch.load(
#         run / "best.pt",
#         map_location=device,
#     )
#     unwrap_model(model).load_state_dict(checkpoint["model"])

#     test, arrays = pass_epoch(
#         model,
#         loaders["test"],
#         device,
#         mean,
#         std,
#         capture=True,
#     )
#     test["best_epoch"] = checkpoint["epoch"]
#     test["smoke_only"] = args.smoke
#     test["target_offsets"] = data.y_offsets.tolist()

#     write_json(run / "test_metrics.json", test)
#     np.savez_compressed(run / "test_predictions.npz", **arrays)

#     return dict(
#         mode=mode,
#         best_epoch=checkpoint["epoch"],
#         **test["overall"],
#         smoke_only=args.smoke,
#     )


# def main(argv=None):
#     args = parser().parse_args(argv)

#     if min(
#         args.epochs,
#         args.batch_size,
#         args.hidden,
#         args.blocks,
#         args.layers,
#         args.context,
#         args.threads,
#         args.gpus,
#     ) < 1:
#         raise ValueError("Training dimensions and epochs must be positive")

#     torch.set_num_threads(args.threads)

#     # 明确实际读取位置，并检查原始数据文件格式。
#     args.raw = args.raw.expanduser().resolve()
#     print(f"原始数据路径：{args.raw}", flush=True)

#     if not args.raw.is_file():
#         raise FileNotFoundError(f"原始数据文件不存在：{args.raw}")

#     print(f"文件大小：{args.raw.stat().st_size} 字节", flush=True)
#     print(f"SHA256：{digest(args.raw)}", flush=True)

#     if not zipfile.is_zipfile(args.raw):
#         with args.raw.open("rb") as f:
#             header = f.read(16)

#         raise ValueError(
#             "文件不是有效的 NPZ/ZIP，可能格式不符或文件损坏。\n"
#             f"路径：{args.raw}\n"
#             f"文件头：{header!r}"
#         )

#     # 先读取全部图，再确定所需历史长度，避免截断图的滞后信息。
# #     with np.load(args.raw, allow_pickle=True) as z:
# #         required = {
# #             "node_ids",
# #             "feature_names",
# #             "data_model_ready",
# #             "target_observed_mask",
# #         }
# #         missing = required - set(z.files)
# #         if missing:
# #             raise ValueError(f"原始数据缺少字段：{sorted(missing)}")

# #         node_ids = z["node_ids"]
# #         feature_names = z["feature_names"]

# #     archive = GraphArchive(
# #         args.graphs,
# #         node_ids,
# #         feature_names,
# #         args.alpha,
# #         args.variables,
# #     )

#     with np.load(args.raw, allow_pickle=True) as z:
#         # 将数据读取到普通的 Python 变量中
#         node_ids = z['node_ids']
#         feature_names = z['feature_names']
        
#         # 如果数据类型是 bytes，解码为 str (加上 isinstance 判断防止空数组报错)
#         if len(node_ids) > 0 and isinstance(node_ids[0], bytes):
#             node_ids = [x.decode('utf-8') for x in node_ids]
        
#         if len(feature_names) > 0 and isinstance(feature_names[0], bytes):
#             feature_names = [x.decode('utf-8') for x in feature_names]
            
#         # 打印一下类型方便调试
#         print(f"Debug - Node ID type: {type(node_ids[0])}, Example: {node_ids[0]}")
        
#         # 使用转换后的变量创建 archive
#         archive = GraphArchive(args.graphs, node_ids, feature_names, args.alpha, args.variables)
#     data = TrafficData(
#         args.data,
#         args.raw,
#         args.variables,
#         max(args.context, archive.bank.shape[1]),
#     )

#     stamp = args.run_name or (
#         time.strftime("%Y%m%d_%H%M%S")
#         + f"_{time.time_ns() % 1000000:06d}"
#     )
#     category = "checks" if args.smoke or args.prepare_only else "runs"
#     directory = args.output / category / stamp
#     directory.mkdir(parents=True, exist_ok=False)

#     source_dir = directory / "source"
#     source_dir.mkdir()
#     for path in Path(__file__).resolve().parent.glob("*.py"):
#         shutil.copy2(path, source_dir / path.name)

#     write_json(
#         directory / "inputs.json",
#         {
#             "raw": {
#                 "path": str(args.raw.resolve()),
#                 "sha256": digest(args.raw),
#             },
#             "splits": {
#                 split: {
#                     "path": str((args.data / (split + ".npz")).resolve()),
#                     "sha256": digest(args.data / (split + ".npz")),
#                 }
#                 for split in ("train", "val", "test")
#             },
#         },
#     )

#     archive.save(directory / "graphs")

#     coverage = {
#         "context": data.context,
#         "graph_count": len(archive.times),
#         "missing_slots": archive.metadata["missing_slots"],
#         "splits": {},
#     }
#     used = set()

#     for split, ts in data.splits.items():
#         ids = archive.ids(
#             ts[:, None] + np.arange(-data.context + 1, 1)
#         )
#         unique = np.unique(ids)
#         used.update(unique[unique > 0].tolist())

#         coverage["splits"][split] = {
#             "samples": len(ts),
#             "first_origin": int(ts[0]),
#             "last_origin": int(ts[-1]),
#             "graph_bank_ids": unique.tolist(),
#             "samples_without_graph_at_origin": int(
#                 (ids[:, -1] == 0).sum()
#             ),
#         }

#     coverage["unused_graph_bank_ids"] = sorted(
#         set(range(1, len(archive.times) + 1)) - used
#     )
#     write_json(directory / "coverage.json", coverage)

#     if coverage["unused_graph_bank_ids"]:
#         raise ValueError(
#             "Some archive graphs never align with this dataset; "
#             "see coverage.json"
#         )

#     print(
#         f"Validated {len(archive.times)} graphs; "
#         f"context={data.context}; output={directory}",
#         flush=True,
#     )

#     if args.prepare_only:
#         return directory

#     modes = (
#         ["causal", "adaptive", "combined"]
#         if args.mode == "all"
#         else [args.mode]
#     )
#     summary = [
#         run_mode(args, data, archive, directory, mode)
#         for mode in modes
#     ]

#     with open(
#         directory / "comparison.csv",
#         "w",
#         newline="",
#         encoding="utf-8",
#     ) as f:
#         writer = csv.DictWriter(f, fieldnames=list(summary[0]))
#         writer.writeheader()
#         writer.writerows(summary)

#     return directory


# if __name__ == "__main__":
#     main()
# ////////////////////////////////////////////////////////////////////////////////
# """Server training entry point. --prepare-only never trains; --smoke is bounded."""

# import argparse
# import csv
# import json
# import random
# import shutil
# import time
# import zipfile
# from pathlib import Path

# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Subset

# from causal_data import GraphArchive, TrafficData, Samples, digest
# from causal_model import DynamicGraphWaveNet


# ROOT = Path(__file__).resolve().parents[2]


# def parser():
#     p = argparse.ArgumentParser(description=__doc__)

#     p.add_argument("--data", type=Path, default=ROOT / "data" / "data1")
#     p.add_argument(
#         "--raw",
#         type=Path,
#         default=ROOT / "data" / "data1" / "full_year_cleaned.npz",
#     )
#     p.add_argument("--graphs", type=Path, default=ROOT / "data" / "per_window")
#     p.add_argument("--output", type=Path, default=ROOT / "result" / "result1")

#     p.add_argument(
#         "--mode",
#         choices=["causal", "adaptive", "combined", "all"],
#         default="all",
#     )
#     p.add_argument("--variables", choices=["all", "flow"], default="all")
#     p.add_argument("--device", default="cuda")
#     p.add_argument(
#         "--gpus",
#         type=int,
#         default=4,
#         help="Number of GPUs to use with DataParallel; ignored on CPU",
#     )
#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--epochs", type=int, default=100)
#     p.add_argument("--patience", type=int, default=20)
#     p.add_argument("--batch-size", type=int, default=32)
#     p.add_argument("--hidden", type=int, default=32)
#     p.add_argument("--blocks", type=int, default=4)
#     p.add_argument("--layers", type=int, default=2)
#     p.add_argument("--context", type=int, default=13)
#     p.add_argument("--learning-rate", type=float, default=0.001)
#     p.add_argument("--weight-decay", type=float, default=0.0001)
#     p.add_argument("--dropout", type=float, default=0.3)
#     p.add_argument("--alpha", type=float, default=0.05)
#     p.add_argument("--threads", type=int, default=2)
#     p.add_argument("--workers", type=int, default=0)
#     p.add_argument("--prepare-only", action="store_true")
#     p.add_argument(
#         "--smoke",
#         action="store_true",
#         help="Only 2 train batches and 2 eval batches per mode",
#     )
#     p.add_argument("--run-name", default=ROOT/'result'/'result1'/'runs'/'20261003_113949_367198')
#     return p


# def seed_all(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True


# def write_json(path, value):
#     Path(path).write_text(
#         json.dumps(value, indent=2, ensure_ascii=False, default=str),
#         encoding="utf-8",
#     )


# def resolve_device(args):
#     """Return the primary device and wrap-ready device list."""
#     if args.device.startswith("cuda"):
#         if not torch.cuda.is_available():
#             raise RuntimeError("--device cuda requested but CUDA is unavailable")

#         count = torch.cuda.device_count()
#         if count < 1:
#             raise RuntimeError("No CUDA devices found")

#         use = min(args.gpus, count)
#         if use < 1:
#             raise ValueError("--gpus must be >= 1 when using CUDA")

#         print(
#             f"CUDA devices available={count}; using {use} GPU(s)",
#             flush=True,
#         )
#         return torch.device("cuda:0"), use

#     return torch.device(args.device), 0


# def wrap_model(model, device, gpu_count):
#     """Wrap with DataParallel when more than one GPU is used."""
#     if gpu_count > 1:
#         model = torch.nn.DataParallel(
#             model,
#             device_ids=list(range(gpu_count)),
#             output_device=0,
#         )

#     return model.to(device)


# def unwrap_model(model):
#     return model.module if isinstance(model, torch.nn.DataParallel) else model


# class Metrics:
#     def __init__(self, horizon):
#         self.sums = np.zeros((horizon, 5), np.float64)

#     def update(self, pred, truth, valid):
#         # 零流量也是有效目标；只有 MAPE 排除零分母。
#         diff = torch.where(
#             valid, pred - truth, torch.zeros_like(pred)
#         ).double()
#         pct = valid & (truth.abs() > 1e-5)

#         vals = [
#             diff.abs().sum((0, 2)),
#             diff.square().sum((0, 2)),
#             valid.sum((0, 2)),
#             torch.where(
#                 pct,
#                 diff.abs() / truth.abs().clamp_min(1e-5),
#                 torch.zeros_like(diff),
#             ).sum((0, 2)),
#             pct.sum((0, 2)),
#         ]

#         self.sums += torch.stack(vals, -1).detach().cpu().numpy()

#     @staticmethod
#     def row(s):
#         a, q, count, pct, pct_count = map(float, s)
#         return {
#             "mae": a / count if count else None,
#             "rmse": (q / count) ** 0.5 if count else None,
#             "mape": pct / pct_count if pct_count else None,
#             "valid_count": int(count),
#             "mape_count": int(pct_count),
#         }

#     def result(self):
#         return {
#             "overall": self.row(self.sums.sum(0)),
#             "by_horizon": [
#                 dict(horizon=i + 1, **self.row(s))
#                 for i, s in enumerate(self.sums)
#             ],
#         }


# def pass_epoch(
#     model,
#     loader,
#     device,
#     mean,
#     std,
#     optimizer=None,
#     capture=False,
# ):
#     training = optimizer is not None
#     model.train(training)

#     metric = Metrics(len(loader.dataset[0][1]))
#     predictions, targets, masks, origins = [], [], [], []

#     with torch.set_grad_enabled(training):
#         for x, y, valid, graph_ids, origin in loader:
#             x, y, valid, graph_ids = (
#                 v.to(device) for v in (x, y, valid, graph_ids)
#             )

#             pred = model(x, graph_ids) * std + mean

#             if not torch.isfinite(pred).all():
#                 raise FloatingPointError("Nonfinite predictions")

#             if training:
#                 if not valid.any():
#                     continue

#                 loss = (pred - y).abs()[valid].mean()
#                 optimizer.zero_grad(set_to_none=True)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(
#                     model.parameters(),
#                     5.0,
#                     error_if_nonfinite=True,
#                 )
#                 optimizer.step()

#             metric.update(pred, y, valid)

#             if capture:
#                 predictions.append(pred.detach().cpu().numpy())
#                 targets.append(y.cpu().numpy())
#                 masks.append(valid.cpu().numpy())
#                 origins.append(origin.numpy())

#     packed = None
#     if capture:
#         packed = {
#             "prediction": np.concatenate(predictions),
#             "target": np.concatenate(targets),
#             "observed": np.concatenate(masks),
#             "origin_index": np.concatenate(origins),
#         }

#     return metric.result(), packed


# def make_loader(data, archive, split, args):
#     dataset = Samples(data, archive, split)

#     if args.smoke:
#         # 跨时间线取样，覆盖后期图，而不只是没有图的初始阶段。
#         indices = np.linspace(
#             0,
#             len(dataset) - 1,
#             min(len(dataset), args.batch_size * 2),
#             dtype=int,
#         )
#         dataset = Subset(dataset, indices.tolist())

#     return DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         shuffle=split == "train",
#         num_workers=args.workers,
#         drop_last=False,
#         generator=torch.Generator().manual_seed(args.seed),
#     )


# def run_mode(args, data, archive, directory, mode):
#     seed_all(args.seed)

#     run = directory / mode
#     # ===== 修改 1：允许目录已存在，支持断点续训 =====
#     run.mkdir(parents=True, exist_ok=True)

#     device, gpu_count = resolve_device(args)
#     model = DynamicGraphWaveNet(
#         data.values.shape[1],
#         data.horizon,
#         data.flow_indices,
#         archive.bank,
#         mode,
#         args.hidden,
#         args.blocks,
#         args.layers,
#         args.dropout,
#     )
#     model = wrap_model(model, device, gpu_count)

#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=args.learning_rate,
#         weight_decay=args.weight_decay,
#     )

#     loaders = {
#         split: make_loader(data, archive, split, args)
#         for split in ("train", "val", "test")
#     }

#     mean = torch.as_tensor(data.output_mean, device=device)
#     std = torch.as_tensor(data.output_std, device=device)

#     config = dict(
#         vars(args),
#         mode=mode,
#         context=data.context,
#         horizon=data.horizon,
#         variables_count=data.values.shape[1],
#         parameters=sum(p.numel() for p in model.parameters()),
#         torch_version=str(torch.__version__),
#         numpy_version=np.__version__,
#         gpu_count=gpu_count,
#     )
#     config = json.loads(json.dumps(config, default=str))
#     write_json(run / "config.json", config)

#     best, stale, history = float("inf"), 0, []
#     epochs = 1 if args.smoke else args.epochs
#     start_epoch = 1

#     # ===== 修改 2：断点续训 —— 若存在 last.pt 则恢复状态 =====
#     last_ckpt_path = run / "last.pt"
#     if not args.smoke and last_ckpt_path.exists():
#         print(
#             f"[{mode}] 检测到断点文件 {last_ckpt_path}，正在恢复训练状态 ...",
#             flush=True,
#         )
#         checkpoint = torch.load(last_ckpt_path, map_location=device)
#         unwrap_model(model).load_state_dict(checkpoint["model"])
#         optimizer.load_state_dict(checkpoint["optimizer"])
#         start_epoch = int(checkpoint["epoch"]) + 1
#         best = float(checkpoint.get("best", float("inf")))
#         stale = int(checkpoint.get("stale", 0))
#         history = list(checkpoint.get("history", []))
#         print(
#             f"[{mode}] 已恢复到 epoch {start_epoch}，"
#             f"历史 best val_MAE={best:.4f}，stale={stale}",
#             flush=True,
#         )
#         if start_epoch > epochs:
#             print(
#                 f"[{mode}] 已完成 {start_epoch - 1} 个 epoch "
#                 f">= 目标 {epochs}，跳过训练直接评估",
#                 flush=True,
#             )
#     else:
#         print(f"[{mode}] 未发现断点文件，从头开始训练", flush=True)
#     # ======================================================

#     for epoch in range(start_epoch, epochs + 1):
#         start = time.time()

#         train, _ = pass_epoch(
#             model, loaders["train"], device, mean, std, optimizer
#         )
#         val, _ = pass_epoch(
#             model, loaders["val"], device, mean, std
#         )

#         score = val["overall"]["mae"]
#         if score is None or not np.isfinite(score):
#             raise ValueError("Validation has no finite observed-target metric")

#         row = dict(
#             epoch=epoch,
#             seconds=time.time() - start,
#             **{"train_" + k: v for k, v in train["overall"].items()},
#             **{"val_" + k: v for k, v in val["overall"].items()},
#         )
#         history.append(row)

#         with open(
#             run / "history.csv", "w", newline="", encoding="utf-8"
#         ) as f:
#             writer = csv.DictWriter(f, fieldnames=list(row))
#             writer.writeheader()
#             writer.writerows(history)

#         print(
#             f'{mode} epoch={epoch} '
#             f'train_MAE={train["overall"]["mae"]:.4f} '
#             f'val_MAE={score:.4f}',
#             flush=True,
#         )

#         if score < best:
#             best, stale = score, 0
#             # ===== 修改 3：best.pt 里也记录 best/stale/history =====
#             torch.save(
#                 {
#                     "model": unwrap_model(model).state_dict(),
#                     "optimizer": optimizer.state_dict(),
#                     "epoch": epoch,
#                     "validation": val,
#                     "config": config,
#                     "best": best,
#                     "stale": stale,
#                     "history": history,
#                     "scaler_mean": torch.from_numpy(data.mean),
#                     "scaler_std": torch.from_numpy(data.std),
#                     "graph_times": torch.from_numpy(archive.times),
#                 },
#                 run / "best.pt",
#             )
#         else:
#             stale += 1

#         # ===== 修改 4：每个 epoch 结束都写 last.pt，供断点续训 =====
#         torch.save(
#             {
#                 "model": unwrap_model(model).state_dict(),
#                 "optimizer": optimizer.state_dict(),
#                 "epoch": epoch,
#                 "validation": val,
#                 "config": config,
#                 "best": best,
#                 "stale": stale,
#                 "history": history,
#                 "scaler_mean": torch.from_numpy(data.mean),
#                 "scaler_std": torch.from_numpy(data.std),
#                 "graph_times": torch.from_numpy(archive.times),
#             },
#             run / "last.pt",
#         )
#         # =========================================================

#         if args.patience and stale >= args.patience:
#             break

#     checkpoint = torch.load(
#         run / "best.pt",
#         map_location=device,
#     )
#     unwrap_model(model).load_state_dict(checkpoint["model"])

#     test, arrays = pass_epoch(
#         model,
#         loaders["test"],
#         device,
#         mean,
#         std,
#         capture=True,
#     )
#     test["best_epoch"] = checkpoint["epoch"]
#     test["smoke_only"] = args.smoke
#     test["target_offsets"] = data.y_offsets.tolist()

#     write_json(run / "test_metrics.json", test)
#     np.savez_compressed(run / "test_predictions.npz", **arrays)

#     return dict(
#         mode=mode,
#         best_epoch=checkpoint["epoch"],
#         **test["overall"],
#         smoke_only=args.smoke,
#     )


# def main(argv=None):
#     args = parser().parse_args(argv)

#     if min(
#         args.epochs,
#         args.batch_size,
#         args.hidden,
#         args.blocks,
#         args.layers,
#         args.context,
#         args.threads,
#         args.gpus,
#     ) < 1:
#         raise ValueError("Training dimensions and epochs must be positive")

#     torch.set_num_threads(args.threads)

#     # 明确实际读取位置，并检查原始数据文件格式。
#     args.raw = args.raw.expanduser().resolve()
#     print(f"原始数据路径：{args.raw}", flush=True)

#     if not args.raw.is_file():
#         raise FileNotFoundError(f"原始数据文件不存在：{args.raw}")

#     print(f"文件大小：{args.raw.stat().st_size} 字节", flush=True)
#     print(f"SHA256：{digest(args.raw)}", flush=True)

#     if not zipfile.is_zipfile(args.raw):
#         with args.raw.open("rb") as f:
#             header = f.read(16)

#         raise ValueError(
#             "文件不是有效的 NPZ/ZIP，可能格式不符或文件损坏。\n"
#             f"路径：{args.raw}\n"
#             f"文件头：{header!r}"
#         )

#     with np.load(args.raw, allow_pickle=True) as z:
#         # 将数据读取到普通的 Python 变量中
#         node_ids = z['node_ids']
#         feature_names = z['feature_names']

#         # 如果数据类型是 bytes，解码为 str (加上 isinstance 判断防止空数组报错)
#         if len(node_ids) > 0 and isinstance(node_ids[0], bytes):
#             node_ids = [x.decode('utf-8') for x in node_ids]

#         if len(feature_names) > 0 and isinstance(feature_names[0], bytes):
#             feature_names = [x.decode('utf-8') for x in feature_names]

#         # 打印一下类型方便调试
#         print(f"Debug - Node ID type: {type(node_ids[0])}, Example: {node_ids[0]}")

#         # 使用转换后的变量创建 archive
#         archive = GraphArchive(args.graphs, node_ids, feature_names, args.alpha, args.variables)

#     data = TrafficData(
#         args.data,
#         args.raw,
#         args.variables,
#         max(args.context, archive.bank.shape[1]),
#     )

#     stamp = args.run_name or (
#         time.strftime("%Y%m%d_%H%M%S")
#         + f"_{time.time_ns() % 1000000:06d}"
#     )
#     category = "checks" if args.smoke or args.prepare_only else "runs"
#     directory = args.output / category / stamp
#     # ===== 修改 5：允许目录已存在，支持用 --run-name 断点续训 =====
#     directory.mkdir(parents=True, exist_ok=True)

#     source_dir = directory / "source"
#     source_dir.mkdir(exist_ok=True)  # 允许已存在
#     for path in Path(__file__).resolve().parent.glob("*.py"):
#         shutil.copy2(path, source_dir / path.name)

#     write_json(
#         directory / "inputs.json",
#         {
#             "raw": {
#                 "path": str(args.raw.resolve()),
#                 "sha256": digest(args.raw),
#             },
#             "splits": {
#                 split: {
#                     "path": str((args.data / (split + ".npz")).resolve()),
#                     "sha256": digest(args.data / (split + ".npz")),
#                 }
#                 for split in ("train", "val", "test")
#             },
#         },
#     )

#     archive.save(directory / "graphs")

#     coverage = {
#         "context": data.context,
#         "graph_count": len(archive.times),
#         "missing_slots": archive.metadata["missing_slots"],
#         "splits": {},
#     }
#     used = set()

#     for split, ts in data.splits.items():
#         ids = archive.ids(
#             ts[:, None] + np.arange(-data.context + 1, 1)
#         )
#         unique = np.unique(ids)
#         used.update(unique[unique > 0].tolist())

#         coverage["splits"][split] = {
#             "samples": len(ts),
#             "first_origin": int(ts[0]),
#             "last_origin": int(ts[-1]),
#             "graph_bank_ids": unique.tolist(),
#             "samples_without_graph_at_origin": int(
#                 (ids[:, -1] == 0).sum()
#             ),
#         }

#     coverage["unused_graph_bank_ids"] = sorted(
#         set(range(1, len(archive.times) + 1)) - used
#     )
#     write_json(directory / "coverage.json", coverage)

#     if coverage["unused_graph_bank_ids"]:
#         raise ValueError(
#             "Some archive graphs never align with this dataset; "
#             "see coverage.json"
#         )

#     print(
#         f"Validated {len(archive.times)} graphs; "
#         f"context={data.context}; output={directory}",
#         flush=True,
#     )

#     if args.prepare_only:
#         return directory

#     modes = (
#         ["causal", "adaptive", "combined"]
#         if args.mode == "all"
#         else [args.mode]
#     )
#     summary = [
#         run_mode(args, data, archive, directory, mode)
#         for mode in modes
#     ]

#     with open(
#         directory / "comparison.csv",
#         "w",
#         newline="",
#         encoding="utf-8",
#     ) as f:
#         writer = csv.DictWriter(f, fieldnames=list(summary[0]))
#         writer.writeheader()
#         writer.writerows(summary)

#     return directory


# if __name__ == "__main__":
#     main()
"""Server training entry point. --prepare-only never trains; --smoke is bounded."""

import argparse
import csv
import json
import random
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from causal_data import GraphArchive, TrafficData, Samples, digest
from causal_model import DynamicGraphWaveNet


ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--data", type=Path, default=ROOT / "data" / "data1")
    p.add_argument(
        "--raw",
        type=Path,
        default="autodl-tmp/GraphWaveNet/data/data1/full_year_cleaned.npz",
    )
    p.add_argument("--graphs", type=Path, default=ROOT / "data" / "per_window")
    p.add_argument("--output", type=Path, default=ROOT / "result" / "result1")

    p.add_argument(
        "--mode",
        choices=["causal", "adaptive", "combined", "all"],
        default="all",
    )
    p.add_argument("--variables", choices=["all", "flow"], default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--gpus",
        type=int,
        default=4,
        help="Number of GPUs to use with DataParallel; ignored on CPU",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument(
        "--min-epochs",
        type=int,
        default=None,
        help=(
            "Minimum epochs before early stopping is allowed. "
            "Default: same as --epochs (run all epochs, no early stop). "
            "Pass 0 to restore plain patience-based early stopping."
        ),
    )
    # ==========================================================================
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--context", type=int, default=13)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Only 2 train batches and 2 eval batches per mode",
    )
    p.add_argument(
        "--run-name",
        default=ROOT / "result" / "result1" / "runs" / "20261003_113949_367198",
    )
    return p


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def resolve_device(args):
    """Return the primary device and wrap-ready device list."""
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")

        count = torch.cuda.device_count()
        if count < 1:
            raise RuntimeError("No CUDA devices found")

        use = min(args.gpus, count)
        if use < 1:
            raise ValueError("--gpus must be >= 1 when using CUDA")

        print(
            f"CUDA devices available={count}; using {use} GPU(s)",
            flush=True,
        )
        return torch.device("cuda:0"), use

    return torch.device(args.device), 0


def wrap_model(model, device, gpu_count):
    """Wrap with DataParallel when more than one GPU is used."""
    if gpu_count > 1:
        model = torch.nn.DataParallel(
            model,
            device_ids=list(range(gpu_count)),
            output_device=0,
        )

    return model.to(device)


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


class Metrics:
    def __init__(self, horizon):
        self.sums = np.zeros((horizon, 5), np.float64)

    def update(self, pred, truth, valid):
        # 零流量也是有效目标；只有 MAPE 排除零分母。
        diff = torch.where(
            valid, pred - truth, torch.zeros_like(pred)
        ).double()
        pct = valid & (truth.abs() > 1e-5)

        vals = [
            diff.abs().sum((0, 2)),
            diff.square().sum((0, 2)),
            valid.sum((0, 2)),
            torch.where(
                pct,
                diff.abs() / truth.abs().clamp_min(1e-5),
                torch.zeros_like(diff),
            ).sum((0, 2)),
            pct.sum((0, 2)),
        ]

        self.sums += torch.stack(vals, -1).detach().cpu().numpy()

    @staticmethod
    def row(s):
        a, q, count, pct, pct_count = map(float, s)
        return {
            "mae": a / count if count else None,
            "rmse": (q / count) ** 0.5 if count else None,
            "mape": pct / pct_count if pct_count else None,
            "valid_count": int(count),
            "mape_count": int(pct_count),
        }

    def result(self):
        return {
            "overall": self.row(self.sums.sum(0)),
            "by_horizon": [
                dict(horizon=i + 1, **self.row(s))
                for i, s in enumerate(self.sums)
            ],
        }


def pass_epoch(
    model,
    loader,
    device,
    mean,
    std,
    optimizer=None,
    capture=False,
):
    training = optimizer is not None
    model.train(training)

    metric = Metrics(len(loader.dataset[0][1]))
    predictions, targets, masks, origins = [], [], [], []

    with torch.set_grad_enabled(training):
        for x, y, valid, graph_ids, origin in loader:
            x, y, valid, graph_ids = (
                v.to(device) for v in (x, y, valid, graph_ids)
            )

            pred = model(x, graph_ids) * std + mean

            if not torch.isfinite(pred).all():
                raise FloatingPointError("Nonfinite predictions")

            if training:
                if not valid.any():
                    continue

                loss = (pred - y).abs()[valid].mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()

            metric.update(pred, y, valid)

            if capture:
                predictions.append(pred.detach().cpu().numpy())
                targets.append(y.cpu().numpy())
                masks.append(valid.cpu().numpy())
                origins.append(origin.numpy())

    packed = None
    if capture:
        packed = {
            "prediction": np.concatenate(predictions),
            "target": np.concatenate(targets),
            "observed": np.concatenate(masks),
            "origin_index": np.concatenate(origins),
        }

    return metric.result(), packed


def make_loader(data, archive, split, args):
    dataset = Samples(data, archive, split)

    if args.smoke:
        # 跨时间线取样，覆盖后期图，而不只是没有图的初始阶段。
        indices = np.linspace(
            0,
            len(dataset) - 1,
            min(len(dataset), args.batch_size * 2),
            dtype=int,
        )
        dataset = Subset(dataset, indices.tolist())

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=split == "train",
        num_workers=args.workers,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
    )


def run_mode(args, data, archive, directory, mode):
    seed_all(args.seed)

    run = directory / mode
    # ===== 修改：允许目录已存在，支持断点续训 =====
    run.mkdir(parents=True, exist_ok=True)

    device, gpu_count = resolve_device(args)
    model = DynamicGraphWaveNet(
        data.values.shape[1],
        data.horizon,
        data.flow_indices,
        archive.bank,
        mode,
        args.hidden,
        args.blocks,
        args.layers,
        args.dropout,
    )
    model = wrap_model(model, device, gpu_count)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    loaders = {
        split: make_loader(data, archive, split, args)
        for split in ("train", "val", "test")
    }

    mean = torch.as_tensor(data.output_mean, device=device)
    std = torch.as_tensor(data.output_std, device=device)

    config = dict(
        vars(args),
        mode=mode,
        context=data.context,
        horizon=data.horizon,
        variables_count=data.values.shape[1],
        parameters=sum(p.numel() for p in model.parameters()),
        torch_version=str(torch.__version__),
        numpy_version=np.__version__,
        gpu_count=gpu_count,
    )
    config = json.loads(json.dumps(config, default=str))
    write_json(run / "config.json", config)

    best, stale, history = float("inf"), 0, []
    epochs = 1 if args.smoke else args.epochs

    # ===== 修改 2：未显式指定 min_epochs 时，默认等于 epochs（跑满不早停） =====
    min_epochs = (
        args.min_epochs if args.min_epochs is not None else epochs
    )
    # ==========================================================================

    start_epoch = 1

    # ===== 断点续训 —— 若存在 last.pt 则恢复状态 =====
    last_ckpt_path = run / "last.pt"
    if not args.smoke and last_ckpt_path.exists():
        print(
            f"[{mode}] 检测到断点文件 {last_ckpt_path}，正在恢复训练状态 ...",
            flush=True,
        )
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best", float("inf")))
        stale = int(checkpoint.get("stale", 0))
        history = list(checkpoint.get("history", []))

        # ===== 修改 3：还没跑到 min_epochs 就把早停计数清零 =====
        if start_epoch <= min_epochs:
            print(
                f"[{mode}] start_epoch={start_epoch} <= "
                f"min_epochs={min_epochs}，重置 stale: {stale} -> 0",
                flush=True,
            )
            stale = 0
        # =========================================================

        print(
            f"[{mode}] 已恢复到 epoch {start_epoch}，"
            f"历史 best val_MAE={best:.4f}，stale={stale}",
            flush=True,
        )
        if start_epoch > epochs:
            print(
                f"[{mode}] 已完成 {start_epoch - 1} 个 epoch "
                f">= 目标 {epochs}，跳过训练直接评估",
                flush=True,
            )
    else:
        print(f"[{mode}] 未发现断点文件，从头开始训练", flush=True)
    # ======================================================

    for epoch in range(start_epoch, epochs + 1):
        start = time.time()

        train, _ = pass_epoch(
            model, loaders["train"], device, mean, std, optimizer
        )
        val, _ = pass_epoch(
            model, loaders["val"], device, mean, std
        )

        score = val["overall"]["mae"]
        if score is None or not np.isfinite(score):
            raise ValueError("Validation has no finite observed-target metric")

        row = dict(
            epoch=epoch,
            seconds=time.time() - start,
            **{"train_" + k: v for k, v in train["overall"].items()},
            **{"val_" + k: v for k, v in val["overall"].items()},
        )
        history.append(row)

        with open(
            run / "history.csv", "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)

        print(
            f'{mode} epoch={epoch} '
            f'train_MAE={train["overall"]["mae"]:.4f} '
            f'val_MAE={score:.4f}',
            flush=True,
        )

        if score < best:
            best, stale = score, 0
            torch.save(
                {
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation": val,
                    "config": config,
                    "best": best,
                    "stale": stale,
                    "history": history,
                    "scaler_mean": torch.from_numpy(data.mean),
                    "scaler_std": torch.from_numpy(data.std),
                    "graph_times": torch.from_numpy(archive.times),
                },
                run / "best.pt",
            )
        else:
            stale += 1

        # 每个 epoch 结束都写 last.pt，供断点续训
        torch.save(
            {
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "validation": val,
                "config": config,
                "best": best,
                "stale": stale,
                "history": history,
                "scaler_mean": torch.from_numpy(data.mean),
                "scaler_std": torch.from_numpy(data.std),
                "graph_times": torch.from_numpy(archive.times),
            },
            run / "last.pt",
        )

        # ===== 修改 4：早停条件加上 min_epochs 约束 =====
        if (
            args.patience
            and stale >= args.patience
            and epoch >= min_epochs
        ):
            break
        # ================================================

    checkpoint = torch.load(
        run / "best.pt",
        map_location=device,
    )
    unwrap_model(model).load_state_dict(checkpoint["model"])

    test, arrays = pass_epoch(
        model,
        loaders["test"],
        device,
        mean,
        std,
        capture=True,
    )
    test["best_epoch"] = checkpoint["epoch"]
    test["smoke_only"] = args.smoke
    test["target_offsets"] = data.y_offsets.tolist()

    write_json(run / "test_metrics.json", test)
    np.savez_compressed(run / "test_predictions.npz", **arrays)

    return dict(
        mode=mode,
        best_epoch=checkpoint["epoch"],
        **test["overall"],
        smoke_only=args.smoke,
    )


def main(argv=None):
    args = parser().parse_args(argv)

    if min(
        args.epochs,
        args.batch_size,
        args.hidden,
        args.blocks,
        args.layers,
        args.context,
        args.threads,
        args.gpus,
    ) < 1:
        raise ValueError("Training dimensions and epochs must be positive")

    torch.set_num_threads(args.threads)

    # 明确实际读取位置，并检查原始数据文件格式。
    args.raw = args.raw.expanduser().resolve()
    print(f"原始数据路径：{args.raw}", flush=True)

    if not args.raw.is_file():
        raise FileNotFoundError(f"原始数据文件不存在：{args.raw}")

    print(f"文件大小：{args.raw.stat().st_size} 字节", flush=True)
    print(f"SHA256：{digest(args.raw)}", flush=True)

    if not zipfile.is_zipfile(args.raw):
        with args.raw.open("rb") as f:
            header = f.read(16)

        raise ValueError(
            "文件不是有效的 NPZ/ZIP，可能格式不符或文件损坏。\n"
            f"路径：{args.raw}\n"
            f"文件头：{header!r}"
        )

    with np.load(args.raw, allow_pickle=True) as z:
        # 将数据读取到普通的 Python 变量中
        node_ids = z['node_ids']
        feature_names = z['feature_names']

        # 如果数据类型是 bytes，解码为 str (加上 isinstance 判断防止空数组报错)
        if len(node_ids) > 0 and isinstance(node_ids[0], bytes):
            node_ids = [x.decode('utf-8') for x in node_ids]

        if len(feature_names) > 0 and isinstance(feature_names[0], bytes):
            feature_names = [x.decode('utf-8') for x in feature_names]

        # 打印一下类型方便调试
        print(f"Debug - Node ID type: {type(node_ids[0])}, Example: {node_ids[0]}")

        # 使用转换后的变量创建 archive
        archive = GraphArchive(args.graphs, node_ids, feature_names, args.alpha, args.variables)

    data = TrafficData(
        args.data,
        args.raw,
        args.variables,
        max(args.context, archive.bank.shape[1]),
    )

    stamp = args.run_name or (
        time.strftime("%Y%m%d_%H%M%S")
        + f"_{time.time_ns() % 1000000:06d}"
    )
    category = "checks" if args.smoke or args.prepare_only else "runs"
    directory = args.output / category / stamp
    # ===== 允许目录已存在，支持用 --run-name 断点续训 =====
    directory.mkdir(parents=True, exist_ok=True)

    source_dir = directory / "source"
    source_dir.mkdir(exist_ok=True)  # 允许已存在
    for path in Path(__file__).resolve().parent.glob("*.py"):
        shutil.copy2(path, source_dir / path.name)

    write_json(
        directory / "inputs.json",
        {
            "raw": {
                "path": str(args.raw.resolve()),
                "sha256": digest(args.raw),
            },
            "splits": {
                split: {
                    "path": str((args.data / (split + ".npz")).resolve()),
                    "sha256": digest(args.data / (split + ".npz")),
                }
                for split in ("train", "val", "test")
            },
        },
    )

    archive.save(directory / "graphs")

    coverage = {
        "context": data.context,
        "graph_count": len(archive.times),
        "missing_slots": archive.metadata["missing_slots"],
        "splits": {},
    }
    used = set()

    for split, ts in data.splits.items():
        ids = archive.ids(
            ts[:, None] + np.arange(-data.context + 1, 1)
        )
        unique = np.unique(ids)
        used.update(unique[unique > 0].tolist())

        coverage["splits"][split] = {
            "samples": len(ts),
            "first_origin": int(ts[0]),
            "last_origin": int(ts[-1]),
            "graph_bank_ids": unique.tolist(),
            "samples_without_graph_at_origin": int(
                (ids[:, -1] == 0).sum()
            ),
        }

    coverage["unused_graph_bank_ids"] = sorted(
        set(range(1, len(archive.times) + 1)) - used
    )
    write_json(directory / "coverage.json", coverage)

    if coverage["unused_graph_bank_ids"]:
        raise ValueError(
            "Some archive graphs never align with this dataset; "
            "see coverage.json"
        )

    print(
        f"Validated {len(archive.times)} graphs; "
        f"context={data.context}; output={directory}",
        flush=True,
    )

    if args.prepare_only:
        return directory

    modes = (
        ["causal", "adaptive", "combined"]
        if args.mode == "all"
        else [args.mode]
    )
    summary = [
        run_mode(args, data, archive, directory, mode)
        for mode in modes
    ]

    with open(
        directory / "comparison.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    return directory


if __name__ == "__main__":
    main()