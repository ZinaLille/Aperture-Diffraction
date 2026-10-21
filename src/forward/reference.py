"""Read-only pure-ASM reference and optional frozen-checkpoint comparison.

All metrics/timing/plots go to an exclusively created NEW directory. No weights
are updated and no existing labels or training-result files are rewritten.
"""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.common.physics import PhysicsConfig, angular_spectrum_raw, postprocess, image_metrics
from src.forward.models import ForwardModel, FraunhoferDetectorLayer
from src.common.torch_physics import AngularSpectrumDetectorLayer
from src.forward.train import GroupData, choose_device, check_dataset, aggregate_rows, write_csv
from src.common.io_utils import new_output, read_json, write_json, sha256
from src.common.timing import StageTimer, synchronize, duration, utc_now


def predict(model, aperture, condition, z, pure):
    return model(aperture, z) if pure else model(aperture, condition, z)[0]


@torch.no_grad()
def evaluate_candidate(model, dataset, device, batch_size, pure=True):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    a, target, condition, ids = next(iter(loader))
    a, condition = a.to(device), condition.to(device)
    z = dataset.z_tensor(ids, device)
    # Warm up this method at the actual shape; excluded from forward timing.
    with StageTimer(device) as warmup:
        predict(model, a, condition, z, pure)
    rows, seconds = [], 0.0
    low, high = float("inf"), -float("inf")
    negatives = above = pixels = 0
    for a, target, condition, ids in loader:
        a, condition = a.to(device), condition.to(device)
        z = dataset.z_tensor(ids, device)
        with StageTimer(device) as forward:
            predicted = predict(model, a, condition, z, pure)
        seconds += forward.seconds
        p = predicted[:, 0].cpu().numpy()
        if not np.isfinite(p).all():
            raise RuntimeError("Nonfinite reference/model output")
        low, high = min(low, float(p.min())), max(high, float(p.max()))
        negatives += int(np.sum(p < -1e-6)); above += int(np.sum(p > 1+1e-6)); pixels += p.size
        for ident, prediction, label in zip(ids.tolist(), p, target[:, 0].numpy()):
            rows.append({"id": ident, **image_metrics(prediction, label)})
    return rows, {"forward_seconds": seconds, "seconds_per_image": seconds/len(dataset),
                  "images_per_second": len(dataset)/max(seconds, 1e-12), "warmup_seconds": warmup.seconds,
                  "batch_size": batch_size, "min_prediction": low, "max_prediction": high,
                  "negative_pixel_pct": negatives/pixels*100, "above_one_pixel_pct": above/pixels*100}


def load_frozen(folder, dataset, device):
    checkpoint = torch.load(folder/"best_model.pt", map_location="cpu", weights_only=False)
    if checkpoint["physics_config"] != dataset.manifest["config"]:
        raise ValueError("Checkpoint and dataset physics configs differ")
    if not np.allclose(checkpoint["condition_mean"], dataset.condition_mean) or not np.allclose(checkpoint["condition_std"], dataset.condition_std):
        raise ValueError("Checkpoint conditioning statistics do not match training split")
    cfg = PhysicsConfig(**checkpoint["physics_config"])
    propagator = checkpoint.get("propagator", "fraunhofer")
    model = ForwardModel(checkpoint["kind"], checkpoint["base_ch"], cfg.log_alpha,
                         propagator=propagator, physics_config=cfg,
                         asm_precision=checkpoint.get("asm_precision", "float32")).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def numerical_check(layer, dataset, device, count=2):
    errors = []
    for ident in dataset.ids[:count]:
        i = int(ident)
        a = np.array(dataset.apertures[i], copy=True)
        z = dataset.metadata[i]["z_m"]
        reference = postprocess(angular_spectrum_raw(a, z, layer.config), layer.config.log_alpha)
        with torch.no_grad():
            value = layer(torch.tensor(a, device=device)[None, None], [z])[0, 0].cpu().numpy()
        errors.append(float(np.mean(np.abs(value-reference))))
    if max(errors) > 1e-5:
        raise ValueError(f"Torch/NumPy ASM mismatch: {errors}")
    return {"ids": dataset.ids[:count].tolist(), "mae": errors, "passed": True, "limit": 1e-5}


@torch.no_grad()
def sample_plot(dataset, candidates, device, folder, nf):
    count = min(3, len(dataset))
    fig, axes = plt.subplots(count, len(candidates)+2,
                             figsize=(3*(len(candidates)+2), 3*count), squeeze=False)
    for row in range(count):
        a, target, cond, ident = dataset[row]
        a, cond = a[None].to(device), cond[None].to(device)
        z = dataset.z_tensor([ident], device)
        images = [(a[0, 0].cpu().numpy(), "Aperture")]
        for name, model, pure in candidates:
            p = predict(model, a, cond, z, pure)[0, 0].cpu().numpy()
            images.append((p, f"{name}\nMAE={np.mean(abs(p-target[0].numpy())):.2e}"))
        images.append((target[0].numpy(), "Label"))
        for col, (value, title) in enumerate(images):
            axes[row, col].imshow(value, cmap="gray" if col == 0 else "inferno", vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=8); axes[row, col].axis("off")
    fig.suptitle(f"NF={nf:g}, {dataset.split_name} samples; common [0,1] display range")
    fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(folder/"samples.png", dpi=110); plt.close(fig)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/forward/v1")
    parser.add_argument("--out", default="results/forward/reference/asm_v1")
    parser.add_argument("--compare-run", type=Path, help="Read existing best checkpoints for a fair new inference pass")
    parser.add_argument("--groups", type=float, nargs="+")
    parser.add_argument("--splits", choices=("val", "test"), nargs="+", default=("val", "test"))
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch-size and cpu-threads must be positive")
    root = Path(args.data).resolve()
    manifest = read_json(root/"manifest.json")
    check_dataset(root, manifest)
    if args.compare_run:
        previous = read_json(args.compare_run/"run_config.json")
        if previous["dataset_manifest_sha256"] != sha256(root/"manifest.json"):
            parser.error("Comparison run used a different dataset manifest")
    groups = manifest["groups"]
    if args.groups:
        if any(not any(abs(g["nf"]-x) < 1e-10 for g in groups) for x in args.groups):
            parser.error("Requested NF group is absent")
        groups = [g for g in groups if any(abs(g["nf"]-x) < 1e-10 for x in args.groups)]
    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    out = new_output(args.out)
    config = PhysicsConfig(**manifest["config"])
    write_json(out/"run_config.json", {"arguments": {**vars(args), "compare_run": str(args.compare_run) if args.compare_run else None},
               "dataset_manifest_sha256": sha256(root/"manifest.json"), "started_at_utc": utc_now(),
               "note": "Same-operator consistency reference, not independent evidence of learned network value.",
               "timing_scope": "CUDA-synchronized forward including intensity postprocess; excludes warmup, loading, transfers, metrics, checkpoint construction and plotting."})
    summaries, checks = [], []
    for group in groups:
        name, nf = group["name"], group["nf"]
        for split in args.splits:
            dataset = GroupData(root, name, split)
            dataset.split_name = split
            folder = out/name/split
            folder.mkdir(parents=True)
            asm = AngularSpectrumDetectorLayer(config, args.precision).to(device)
            far = FraunhoferDetectorLayer(config.log_alpha).to(device)
            check = numerical_check(asm, dataset, device)
            checks.append({"nf": nf, "split": split, **check})
            candidates = [("fraunhofer_only", far, True), ("angular_spectrum_only", asm, True)]
            if args.compare_run:
                for kind in ("physics", "baseline"):
                    path = args.compare_run/name/kind
                    if not (path/"best_model.pt").exists():
                        raise FileNotFoundError(f"Comparison checkpoint missing: {path}")
                    candidates.append(("trained_"+kind, load_frozen(path, dataset, device), False))
            for method, model, pure in candidates:
                rows, timing = evaluate_candidate(model, dataset, device, args.batch_size, pure)
                metrics = aggregate_rows(rows)
                write_csv(folder/(method+"_per_sample.csv"), rows)
                item = {"nf": nf, "split": split, "method": method, "samples": len(dataset),
                        **metrics, **timing}
                summaries.append(item)
                print(f"NF={nf:g} {split} {method}: MAE={metrics['mae']:.3e}, Corr={metrics['corr']:.6f}, forward={timing['seconds_per_image']*1000:.2f} ms/image", flush=True)
            sample_plot(dataset, candidates, device, folder, nf)
            write_csv(out/"summary.csv", summaries)
            write_json(out/"implementation_checks.json", checks)
    test_rows = [r for r in summaries if r["split"] == "test"]
    if test_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for method in dict.fromkeys(r["method"] for r in test_rows):
            selected = [r for r in test_rows if r["method"] == method]
            ax.loglog([r["nf"] for r in selected], [max(r["mae"], 1e-12) for r in selected], "o-", label=method)
        ax.set(xlabel="Fresnel number", ylabel="Test MAE (floor 1e-12 for display)", title="Pure ASM reference; no training")
        ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(out/"test_mae_comparison.png", dpi=140); plt.close(fig)
    synchronize(device)
    elapsed = time.perf_counter()-started
    write_json(out/"timing.json", {"wall_seconds": elapsed, "finished_at_utc": utc_now(),
               "forward_scope": "See run_config.json; forward timings cannot be compared to old training durations."})
    lines = ["# 纯角谱参照评估", "", "所有数据和旧权重只读；本次未训练网络。", "",
             "| NF | 方法 | 测试MAE | Corr | 仅前向ms/图 |", "|---|---|---:|---:|---:|"]
    for row in test_rows:
        lines.append(f"| {row['nf']:g} | {row['method']} | {row['mae']:.6g} | {row['corr']:.6f} | {row['seconds_per_image']*1000:.3f} |")
    lines += ["", "近场标签本就由角谱参考生成，极小误差主要验证实现一致性，不证明神经网络学习收益。",
              "远场标签采用点采样FFT；纯角谱仍包含像素面积积分，存在小差异是可预期的。因此训练新增hybrid模式，远场保留FFT。",
              "前向计时在同一设备/批量下重新测量旧模型，CUDA边界同步，排除首次预热、传输、指标与绘图。ASM精度设置见run_config，比较速度时须考虑精度差异。",
              "旧训练未记录时间，不能从现有日志恢复其真实训练耗时。新计时仅影响后续新训练。", "", "![对比](test_mae_comparison.png)"]
    (out/"report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Reference complete: {out}; total {duration(elapsed)}")


if __name__ == "__main__":
    main()
