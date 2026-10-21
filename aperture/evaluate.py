"""Evaluate trained models, pure-physics references and the independent RS1 audit.

Usage:
    python -m aperture.evaluate --data data/forward/v1 --run runs/forward/all_v1 --out results/forward/eval_v1
    python -m aperture.evaluate --rs1 --data data/forward/v1 --run runs/forward/all_v1 \
        --asm-run runs/forward/asm_nf3_nf5_e10_v1 --out results/forward/rs1_v1
"""

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Allow `python aperture/generate.py` and similar direct runs.
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT))
    os.chdir(_ROOT)

import argparse
import hashlib
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aperture.apertures import STRATEGIES, make_aperture
from aperture.models import ForwardModel, FraunhoferDetectorLayer
from aperture.physics import (PhysicsConfig, AngularSpectrumDetectorLayer,
                      check_nf, group_name, geometry, postprocess, image_metrics,
                      fraunhofer_raw, angular_spectrum_raw,
                      rayleigh_sommerfeld_points)
from aperture.train import GroupData, choose_device, check_dataset, aggregate_rows, write_csv
from aperture.utils import (new_output, read_json, write_json, sha256,
                    StageTimer, synchronize, duration, utc_now)




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


def main_reference():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/forward/v1")
    parser.add_argument("--out", default="results/forward/reference/asm_v1")
    parser.add_argument("--run", type=Path, help="Trained run to compare; omit for pure-physics only")
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
    if args.run:
        previous = read_json(args.run/"run_config.json")
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
    write_json(out/"run_config.json", {"arguments": {**vars(args), "run": str(args.run) if args.run else None},
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
            if args.run:
                for kind in ("physics", "baseline"):
                    path = args.run/name/kind
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


def aperture_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def fraunhofer_cell_raw(aperture, z, config):
    """Analytical far-field integral of the same constant square pixel cells.

    Each pixel contributes its point-source Fourier term times sinc(fx*dx)
    sinc(fy*dx). On the common detector grid f*dx=(j-N/2)/N.
    """
    n = config.grid
    response = np.sinc((np.arange(n)-n//2)/n)**2
    return fraunhofer_raw(aperture, z, config)*response[:,None]*response[None,:]


def fresh_apertures(config, old, per_family, seed):
    forbidden = {aperture_hash(a) for a in old}
    original = set(forbidden)
    rng = np.random.RandomState(seed)
    arrays, metadata = [], []
    for family in STRATEGIES:
        for replicate in range(per_family):
            for attempt in range(1000):
                a = make_aperture(family, rng, config.grid)
                digest = aperture_hash(a)
                if digest not in forbidden:
                    break
            else:
                raise RuntimeError("Could not obtain a new, nonempty, unique aperture")
            forbidden.add(digest)
            metadata.append({"id": len(arrays), "strategy": family, "replicate": replicate,
                             "sha256": digest, "matches_old_data": digest in original})
            arrays.append(a)
    return np.stack(arrays), metadata


def detector_indices(n, maps):
    """Fixed central grid/rings plus symmetric candidate-peak neighbourhoods.

    Candidate peaks from BOTH physics methods are used; no point is removed on
    the basis of errors. All indices are shared across all quadrature levels.
    """
    centre = n//2
    selected = {(centre+y, centre+x) for y in range(-2, 3) for x in range(-2, 3)}
    radii = sorted(set(min(r, n//2-2) for r in (4, 8, 16, n//8, n//4, 7*n//16)))
    for radius in radii:
        for dy, dx in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)):
            selected.add((centre+dy*radius, centre+dx*radius))
    for value in maps:
        py, px = np.unravel_index(np.argmax(value), value.shape)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if 0 <= py+dy < n and 0 <= px+dx < n:
                    selected.add((py+dy, px+dx))
    return np.asarray(sorted(selected), dtype=np.int64)


def common_log(raw, scale, alpha):
    # No per-method peak normalization, clipping, or removal of overshoots.
    return np.log1p(alpha*np.asarray(raw)/scale)/np.log1p(alpha)


def physical_metrics(predicted, reference, alpha=100.0):
    scale = max(float(np.max(reference)), 1e-30)
    error = np.abs(predicted-reference)/scale
    return {"linear_mae_ref_peak": float(error.mean()),
            "linear_max_ref_peak": float(error.max()),
            "common_log_mae": float(np.mean(np.abs(common_log(predicted, scale, alpha)
                                                     -common_log(reference, scale, alpha))))}


def integrate_reference(a, z, xy, config, max_level, max_change, log_change):
    values, steps = [], []
    level = 2
    passed = False
    while level <= max_level:
        started = time.perf_counter()
        value = rayleigh_sommerfeld_points(a, z, xy, config, oversample=level)
        if not np.isfinite(value).all() or value.max() <= 0:
            raise ValueError("Nonfinite or empty RS reference")
        step = {"oversample": level, "seconds": time.perf_counter()-started,
                "max_change_ref_peak": None, "mean_change_ref_peak": None,
                "common_log_change": None, "decreasing": None, "passed": False}
        if values:
            difference = physical_metrics(values[-1], value, config.log_alpha)
            step.update(max_change_ref_peak=difference["linear_max_ref_peak"],
                        mean_change_ref_peak=difference["linear_mae_ref_peak"],
                        common_log_change=difference["common_log_mae"])
            # Require two refinements (2->4->8) and a decreasing change.
            decreasing = (len(steps) >= 2 and (step["max_change_ref_peak"] <=
                          0.8*steps[-1]["max_change_ref_peak"] or step["max_change_ref_peak"] < 1e-11))
            passed = bool(decreasing and step["max_change_ref_peak"] <= max_change
                          and step["common_log_change"] <= log_change)
            step.update(decreasing=decreasing, passed=passed)
        values.append(value)
        steps.append(step)
        if passed:
            break
        level *= 2
    return np.stack(values), steps, passed


@torch.no_grad()
def frozen_predictions(models, aperture, geo, dataset, device):
    a = torch.tensor(aperture, device=device)[None, None]
    raw_conditions = np.log([geo["z_m"], geo["detector_pitch_m"]])
    condition = torch.tensor((raw_conditions-dataset.condition_mean)/dataset.condition_std,
                             dtype=torch.float32, device=device)[None]
    z = torch.tensor([geo["z_m"]], dtype=torch.float64, device=device)
    return {name: model(a, condition, z)[0][0,0].cpu().numpy() for name, model in models.items()}


def aggregate(case_rows, comparisons):
    rows = []
    for nf in sorted(set(r["nf"] for r in case_rows)):
        for method in dict.fromkeys(r["method"] for r in case_rows):
            selected = [r for r in case_rows if r["nf"] == nf and r["method"] == method and r["rs_converged"]]
            if not selected:
                continue
            row = {"nf": nf, "method": method, "converged_cases": len(selected)}
            for metric in ("linear_mae_ref_peak", "linear_max_ref_peak", "common_log_mae"):
                row[metric] = float(np.mean([r[metric] for r in selected]))
            rows.append(row)
    wins = []
    for nf in sorted(set(r["nf"] for r in comparisons)):
        subset = [r for r in comparisons if r["nf"] == nf and r["rs_converged"]]
        wins.append({"nf": nf, "cases": len(subset),
                     "asm_better_cases": sum(r["asm_better"] for r in subset),
                     "asm_better_than_legacy_fft_cases": sum(r["asm_better_than_legacy_fft"] for r in subset),
                     "margin_over_5x_ref_change_cases": sum(r["resolved_margin"] for r in subset)})
    return rows, wins


def figures(out, apertures, meta, previews, summaries):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in dict.fromkeys(r["method"] for r in summaries):
        rows = [r for r in summaries if r["method"] == method]
        ax.loglog([r["nf"] for r in rows], [max(r["linear_mae_ref_peak"], 1e-12) for r in rows], "o-", label=method)
    ax.set(xlabel="Fresnel number", ylabel="Mean sampled absolute error / RS sampled peak",
           title="Independent RS1 audit (converged cases only)")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out/"physics_comparison.png", dpi=150); plt.close(fig)
    for nf, cases in previews.items():
        fig, axes = plt.subplots(8, 3, figsize=(13, 24), squeeze=False)
        for row, case in enumerate(cases):
            ident, indices, raw, ref, fft, asm = case
            axes[row,0].imshow(apertures[ident], cmap="gray", vmin=0, vmax=1)
            axes[row,0].set_title(meta[ident]["strategy"]); axes[row,0].axis("off")
            axes[row,1].imshow(postprocess(raw), cmap="inferno", vmin=0, vmax=1)
            axes[row,1].scatter(indices[:,1], indices[:,0], s=6, facecolors="none", edgecolors="cyan", linewidths=.4)
            axes[row,1].set_title("ASM image; RS sample locations"); axes[row,1].axis("off")
            scale = ref.max()
            axes[row,2].semilogy(np.maximum(abs(fft-ref)/scale, 1e-12), label="Fraunhofer")
            axes[row,2].semilogy(np.maximum(abs(asm-ref)/scale, 1e-12), label="ASM c2")
            axes[row,2].set(xlabel="Sample index (row-major; not a spatial cut)", ylabel="Abs error / RS sampled peak")
            axes[row,2].legend(fontsize=8); axes[row,2].grid(alpha=.2)
        fig.suptitle(f"NF={nf:g}; first fresh aperture in each family", fontsize=16)
        fig.tight_layout(rect=(0,0,1,.98)); fig.savefig(out/f"{group_name(nf)}_examples.png", dpi=110); plt.close(fig)


def main_rs1():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/forward/v1"))
    parser.add_argument("--run", type=Path, default=Path("runs/forward/all_v1"))
    parser.add_argument("--asm-run", type=Path, default=Path("runs/forward/asm_nf3_nf5_e10_v1"))
    parser.add_argument("--out", default="results/forward/checks/rs1_audit_v1")
    parser.add_argument("--groups", type=float, nargs="+", default=(0.03, 3, 5))
    parser.add_argument("--per-family", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--max-rs-level", type=int, choices=(8,16,32), default=16)
    parser.add_argument("--max-ref-change", type=float, default=5e-4)
    parser.add_argument("--max-log-change", type=float, default=2e-4)
    parser.add_argument("--device", choices=("cpu","cuda","auto"), default="auto")
    args = parser.parse_args()
    if args.per_family < 1 or min(args.max_ref_change, args.max_log_change) <= 0:
        parser.error("positive sample count and tolerances required")
    started = time.perf_counter()
    nf_values = check_nf(args.groups)
    manifest = read_json(args.data/"manifest.json")
    check_dataset(args.data, manifest)
    for run in (args.run, args.asm_run):
        if read_json(run/"run_config.json")["dataset_manifest_sha256"] != sha256(args.data/"manifest.json"):
            raise ValueError("Frozen run and audit source dataset disagree")
    if any(nf not in [g["nf"] for g in manifest["groups"]] for nf in nf_values):
        raise ValueError("Audit group must have a matching original data group")
    cfg = PhysicsConfig(**manifest["config"])
    old = np.load(args.data/"apertures.npy", mmap_mode="r")
    apertures, meta = fresh_apertures(cfg, old, args.per_family, args.seed)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    out = new_output(args.out)
    (out/"cases").mkdir()
    np.save(out/"audit_apertures.npy", apertures)
    write_json(out/"apertures.json", meta)
    provenance = {"arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  "config": cfg.to_dict(), "started_at_utc": utc_now(), "device": str(device),
                  "dataset_manifest_sha256": sha256(args.data/"manifest.json"),
                  "original_samples_excluded": len(old), "fresh_aperture_count": len(apertures),
                  "point_selection": "fixed central grid and rings; union of 3x3 around FFT and ASM peaks; shared across solvers/levels",
                  "reference": "direct RS1 midpoint quadrature, including 1/(kr), complex128 numpy; independent of FFT propagation",
                  "matched_cell_control": "fraunhofer_cells analytically integrates square constant pixels via the sinc envelope; asm_better refers to this control",
                  "code_sha256": {name: sha256(Path(__file__).parent/name) for name in ("evaluate.py","train.py","models.py","physics.py","utils.py")},
                  "primary_normalization": "one max(RS at all sampled points) shared by every physical method; no individual peak fit",
                  "convergence": "2->4->8 then adaptive 16/32 if requested; changes are empirical indicators, not rigorous error bounds",
                  "network_scope": "auxiliary normalized-log sampled errors; reference full-image peak estimated from sampled points, no raw intensity recovery",
                  "checkpoint_files": []}
    write_json(out/"run_config.json", provenance)
    case_rows, network_rows, convergence, comparisons, previews = [], [], [], [], {}
    for nf in nf_values:
        dataset = GroupData(args.data, group_name(nf), "val")
        models = {}
        for method, run, kind in (("fft_residual",args.run,"physics"),
                                  ("baseline",args.run,"baseline"),
                                  ("asm_residual",args.asm_run,"physics")):
            folder = run/group_name(nf)/kind
            if method == "asm_residual" and not (folder/"best_model.pt").exists():
                continue
            models[method] = load_frozen(folder, dataset, device)
            provenance["checkpoint_files"].append({"nf": nf, "method": method,
                      "path": str((folder/"best_model.pt").resolve()), "sha256": sha256(folder/"best_model.pt"),
                      "selection": read_json(folder/"selection.json")})
        write_json(out/"run_config.json", provenance)
        previews[nf] = []
        for ident, a in enumerate(apertures):
            case_started = time.perf_counter()
            item = meta[ident]
            geo = geometry(a, nf, cfg)
            far = fraunhofer_raw(a, geo["z_m"], cfg)
            cells = fraunhofer_cell_raw(a, geo["z_m"], cfg)
            asm = angular_spectrum_raw(a, geo["z_m"], cfg)
            refined = angular_spectrum_raw(a, geo["z_m"], replace(cfg, oversample=cfg.oversample*2))
            indices = detector_indices(cfg.grid, (far, cells, asm))
            xy = (indices[:,::-1]-cfg.grid//2)*geo["detector_pitch_m"]
            rs, steps, passed = integrate_reference(a, geo["z_m"], xy, cfg, args.max_rs_level,
                                                    args.max_ref_change, args.max_log_change)
            ref, last = rs[-1], steps[-1]
            y, x = indices.T
            base = {"nf": nf, "id": ident, "strategy": item["strategy"], "replicate": item["replicate"],
                    "points": len(indices), "rs_converged": passed, "rs_level": last["oversample"]}
            for step in steps:
                convergence.append({**base, **step})
            scores = {}
            point_predictions = {}
            radius = np.max(abs(indices-cfg.grid//2), axis=1)
            for method, full in (("fraunhofer",far),("fraunhofer_cells",cells),("asm_c2",asm),("asm_c4",refined)):
                values = full[y,x]
                point_predictions[method] = values
                scores[method] = physical_metrics(values, ref, cfg.log_alpha)
                row = {**base, "method": method, **scores[method]}
                for name, mask in (("central_r_le_4",radius<=4),("middle_r_5_16",(radius>4)&(radius<=16)),("outer_r_gt_16",radius>16)):
                    row[name+"_linear_mae_ref_peak"] = float(np.mean(abs(values[mask]-ref[mask]))/ref.max()) if mask.any() else None
                case_rows.append(row)
            gap = scores["fraunhofer_cells"]["linear_mae_ref_peak"]-scores["asm_c2"]["linear_mae_ref_peak"]
            legacy_gap = scores["fraunhofer"]["linear_mae_ref_peak"]-scores["asm_c2"]["linear_mae_ref_peak"]
            comparisons.append({**base, "matched_cells_minus_asm_mae": gap, "legacy_fft_minus_asm_mae":legacy_gap,
                                "reference_last_mean_change": last["mean_change_ref_peak"],
                                "asm_better": gap > 0,
                                "asm_better_than_legacy_fft": legacy_gap > 0,
                                "resolved_margin": bool(gap > 5*last["mean_change_ref_peak"]),
                                "seconds": time.perf_counter()-case_started})
            normalized = {"fraunhofer":postprocess(far,cfg.log_alpha), "asm_c2":postprocess(asm,cfg.log_alpha)}
            normalized.update(frozen_predictions(models, a, geo, dataset, device))
            log_ref = common_log(ref, ref.max(), cfg.log_alpha)
            for method, value in normalized.items():
                if not np.isfinite(value).all():
                    raise ValueError("Nonfinite model output")
                network_rows.append({**base, "method": method,
                                     "sampled_normalized_log_mae": float(np.mean(abs(value[y,x]-log_ref))),
                                     "full_prediction_min": float(value.min()), "full_prediction_max": float(value.max())})
            case_path = out/"cases"/f"{group_name(nf)}_a{ident:03d}.npz"
            np.savez_compressed(case_path, indices_yx=indices, coordinates_xy_m=xy, rs_intensities=rs,
                                rs_levels=np.array([s["oversample"] for s in steps]),
                                **point_predictions, **{"log_"+k:v[y,x] for k,v in normalized.items()})
            write_json(case_path.with_suffix(".json"), {**base, **geo, "sampled_rs_peak": float(ref.max()),
                       "aperture_sha256": item["sha256"], "steps": steps})
            if item["replicate"] == 0:
                previews[nf].append((ident, indices, asm, ref, far[y,x], asm[y,x]))
            for filename, rows in (("case_metrics.csv",case_rows),("model_metrics_auxiliary.csv",network_rows),
                                   ("convergence.csv",convergence),("comparisons.csv",comparisons)):
                write_csv(out/filename, rows)
            write_json(out/"progress.json", {"completed_cases":len(comparisons), "total_cases":len(apertures)*len(nf_values),
                                            "elapsed_seconds":time.perf_counter()-started})
            print(f"NF={nf:g} {ident+1}/{len(apertures)} {item['strategy']}: RS c{last['oversample']} "
                  f"converged={passed}, FFT={scores['fraunhofer']['linear_mae_ref_peak']:.3e}, "
                  f"ASM={scores['asm_c2']['linear_mae_ref_peak']:.3e}, ref_change={last['mean_change_ref_peak']:.2e}, "
                  f"{time.perf_counter()-case_started:.1f}s", flush=True)
        del models
    summaries, wins = aggregate(case_rows, comparisons)
    write_csv(out/"summary.csv", summaries)
    write_json(out/"win_counts.json", wins)
    write_json(out/"run_config.json", provenance)
    figures(out, apertures, meta, previews, summaries)
    complete = {"status":"completed", "cases":len(comparisons), "converged_cases":sum(r["rs_converged"] for r in comparisons),
                "wall_seconds":time.perf_counter()-started, "finished_at_utc":utc_now(),
                "all_fresh":not any(m["matches_old_data"] for m in meta)}
    write_json(out/"completion.json", complete)
    lines = ["# 独立 RS1 传播审计", "", "仅第一阶段正向评估；旧数据与权重只读，没有训练。", "",
             f"新增 {len(apertures)} 个光阑（八类各 {args.per_family} 个），与原 {len(old)} 个输入逐一核对哈希无重复。",
             f"共 {len(comparisons)} 个传播案例，{complete['converged_cases']} 个通过积分收敛门槛。", "",
             "## 纯物理结果", "", "主指标：抽样位置平均绝对强度误差 / RS抽样峰值。所有物理方法共用同一分母，保留强度尺度误差。",
             "不是原测试集全图MAE，不能与旧MAE直接对比。",
             "fraunhofer是旧点采样FFT；fraunhofer_cells加入解析sinc像素面积因子，使光阑像素解释与RS1一致；ASM c2/c4分别为2/4倍子像素。", "",
             "| NF | 方法 | 通过收敛案例 | 平均线性误差/RS峰值 | 共同尺度log误差 |", "|---|---|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['nf']:g} | {row['method']} | {row['converged_cases']} | {row['linear_mae_ref_peak']:.6g} | {row['common_log_mae']:.6g} |")
    lines += ["", "## 胜出与参考收敛", "", "下表比较ASM c2与已对齐像素面积的fraunhofer_cells。", "",
              "| NF | 收敛案例 | ASM误差更低 | 改善大于5倍参考末次变化 |", "|---|---:|---:|---:|"]
    for row in wins:
        lines.append(f"| {row['nf']:g} | {row['cases']} | {row['asm_better_cases']} | {row['margin_over_5x_ref_change_cases']} |")
    lines += ["", "参考使用RS1直接积分（包含1/(kr)项），从2、4、8倍子像素依次加密，必要时继续加密。",
              f"验收要求：末次最大强度变化/RS峰值≤{args.max_ref_change:g}、共同尺度log平均变化≤{args.max_log_change:g}，且变化下降。",
              "末次变化是经验收敛指标，不是严格误差上界；5倍门槛用于保守区分改善幅度与参考变化。",
              "有限抽样不保证全图精度，也不证明真实实验准确性。中心/中间/外围分区按像素半径定义，不等同于所有形状的主瓣/旁瓣。", "",
              "远场比较还包含点采样FFT与像素单元积分的差异，不能把所有差别归为远场近似误差。", "",
              "## 已有网络的辅助比较", "", "见model_metrics_auxiliary.csv。使用原权重的归一化log图，不更新参数、不裁剪输出。",
              "RS全图峰值只能由抽样点估计，点集包含两种物理预测峰值邻域。此项只供形状/归一化输出辅助检查，不能用于绝对强度结论。",
              "纯物理共同尺度指标与网络归一化指标必须分别解释。没有NF=0.03的角谱训练权重，因此该组不含角谱残差模型。", "",
              "![物理对比](physics_comparison.png)", "",
              f"总耗时 {complete['wall_seconds']:.2f} 秒。每个案例保留实际坐标、各积分级别强度、预测、收敛记录与耗时。"]
    (out/"report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Audit complete: {out}; {complete}", flush=True)


def main():
    if "--rs1" in sys.argv[1:]:
        sys.argv.remove("--rs1")
        main_rs1()
    else:
        main_reference()


if __name__ == "__main__":
    main()
