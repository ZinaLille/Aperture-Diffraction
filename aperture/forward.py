"""Stage-one pipeline: propagation validation, data generation and training.

Usage:
    python -m aperture.forward validate --out results/forward/checks/v1
    python -m aperture.forward generate --out data/forward/v1 ...
    python -m aperture.forward train --data data/forward/v1 ...
"""
import argparse
import csv
import random
import sys
import time
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .apertures import STRATEGIES, make_aperture, split_ids
from .models import ForwardModel, FraunhoferDetectorLayer, Loss
from .physics import (PhysicsConfig, DEFAULT_NF, AngularSpectrumDetectorLayer,
                      check_nf, group_name, geometry, postprocess,
                      fraunhofer_raw, angular_spectrum_raw,
                      rayleigh_sommerfeld_points, image_metrics)
from .utils import (new_output, read_json, write_json, sha256,
                    StageTimer, synchronize, duration, utc_now)

matplotlib.use("Agg")


def standard_apertures(n):
    yy, xx = np.indices((n, n))
    circle = ((xx-n//2)**2+(yy-n//2)**2 <= (n//8)**2).astype(np.float32)
    slit = np.zeros((n, n), np.float32)
    width = max(2, n//32)
    slit[n//4:3*n//4, n//2-width//2:n//2+width//2] = 1
    double = np.zeros_like(slit)
    for center in (n//2-n//10, n//2+n//10):
        double[n//4:3*n//4, center-width//2:center+width//2] = 1
    grating = np.zeros_like(slit)
    period = max(4, n//16)
    for start in range(n//4, 3*n//4, period):
        grating[n//4:3*n//4, start:start+period//2] = 1
    return {"circle": circle, "single_slit": slit, "double_slit": double, "grating": grating}


def validate(config, nf_values, out):
    nf_values = check_nf(nf_values)
    rows, checks = [], []
    standards = standard_apertures(config.grid)
    fig, axes = plt.subplots(len(standards), len(nf_values),
                             figsize=(3*len(nf_values), 3*len(standards)), squeeze=False)
    for row, (name, a) in enumerate(standards.items()):
        for col, nf in enumerate(nf_values):
            geo = geometry(a, nf, config)
            far = postprocess(fraunhofer_raw(a, geo["z_m"], config), config.log_alpha)
            ref = postprocess(angular_spectrum_raw(a, geo["z_m"], config), config.log_alpha)
            metric = image_metrics(far, ref)
            rows.append({"shape": name, "nf": nf, **geo, **metric})
            axes[row, col].imshow(ref, cmap="inferno", vmin=0, vmax=1)
            axes[row, col].set_title(f"{name}, NF={nf:g}\nFFT/ref MAE={metric['mae']:.2e}", fontsize=9)
            axes[row, col].axis("off")
            if nf < config.far_threshold:
                # Includes finite pixel-cell vs point-sampled FFT differences.
                checks.append({"check": "far_agreement", "shape": name, "nf": nf,
                               "value": metric["mae"], "limit": 0.005,
                               "passed": metric["mae"] < 0.005})
        print(f"Validated {name}", flush=True)
    fig.tight_layout()
    fig.savefig(out/"propagation_grid.png", dpi=110)
    plt.close(fig)
    # Reference convergence: do not mistake a single ASM discretization for truth.
    a = standards["circle"]
    for nf in (0.1, 1.0, 5.0):
        geo = geometry(a, nf, config)
        raw = angular_spectrum_raw(a, geo["z_m"], config)
        base = postprocess(raw, config.log_alpha)
        for label, other in (
            ("padding", replace(config, pad_factor=config.pad_factor*2)),
            ("refinement", replace(config, oversample=config.oversample*2)),
        ):
            refined = postprocess(angular_spectrum_raw(a, geo["z_m"], other), config.log_alpha)
            value = image_metrics(base, refined)["mae"]
            checks.append({"check": label, "shape": "circle", "nf": nf,
                           "value": value, "limit": 0.002, "passed": value < 0.002})
        ids = np.array([[config.grid//2, config.grid//2],
                        [config.grid//2, config.grid//2+1],
                        [config.grid//2, config.grid//2+3],
                        [config.grid//2+3, config.grid//2+3]])
        points = (ids[:, ::-1]-config.grid//2)*geo["detector_pitch_m"]
        direct = rayleigh_sommerfeld_points(a, geo["z_m"], points, config,
                                           oversample=config.oversample*2)
        sampled = raw[ids[:, 0], ids[:, 1]]
        value = float(np.max(np.abs(direct-sampled))/max(float(direct.max()), 1e-30))
        checks.append({"check": "independent_RS1", "shape": "circle", "nf": nf,
                       "value": value, "limit": 0.02, "passed": value < 0.02})
        print(f"Convergence and independent RS check NF={nf:g}", flush=True)
    # Include the switch point on BOTH methods. Differences are reported,
    # not hidden by splicing two plots together.
    switch = []
    for name, a in standards.items():
        geo = geometry(a, config.far_threshold, config)
        error = image_metrics(postprocess(fraunhofer_raw(a, geo["z_m"], config), config.log_alpha),
                              postprocess(angular_spectrum_raw(a, geo["z_m"], config), config.log_alpha))["mae"]
        switch.append({"shape": name, "nf": config.far_threshold, "mae": error})
        checks.append({"check": "switch_agreement", "shape": name, "nf": config.far_threshold,
                       "value": error, "limit": 0.005, "passed": error < 0.005})
    with (out/"propagation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in standards:
        subset = [r for r in rows if r["shape"] == name]
        ax.plot([r["nf"] for r in subset], [r["mae"] for r in subset], "o-", label=name)
    ax.set(xscale="log", xlabel="Fresnel number (D = occupied bounding-box width)",
           ylabel="MAE: Fraunhofer vs reference", title="Propagation approximation check")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out/"approximation_error.png", dpi=130)
    plt.close(fig)
    report = {"schema": 1, "config": config.to_dict(), "nf_values": list(nf_values),
              "reference": "scaled ASM: exact angular dispersion correction + scaled Fresnel evaluation",
              "checks": checks, "switch": switch,
              "passed": all(c["passed"] for c in checks),
              "limits_note": "Numerical acceptance criteria for this experiment; not universal physics boundaries."}
    write_json(out/"validation.json", report)
    return report


def add_physics_args(parser):
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--pitch-um", type=float, default=10.0)
    parser.add_argument("--wavelength-nm", type=float, default=532.0)
    parser.add_argument("--far-threshold", type=float, default=0.1)
    parser.add_argument("--oversample", type=int, default=2)
    parser.add_argument("--pad-factor", type=int, default=2)
    parser.add_argument("--nf", type=float, nargs="+", default=DEFAULT_NF)


def config_from_args(args):
    return PhysicsConfig(grid=args.grid, input_pitch_m=args.pitch_um*1e-6,
                         wavelength_m=args.wavelength_nm*1e-9, far_threshold=args.far_threshold,
                         oversample=args.oversample, pad_factor=args.pad_factor)


def main_validate():
    parser = argparse.ArgumentParser(description=__doc__)
    add_physics_args(parser)
    parser.add_argument("--out", default="results/forward/checks/v1")
    args = parser.parse_args()
    config, nf = config_from_args(args), check_nf(args.nf)
    out = new_output(args.out)
    report = validate(config, nf, out)
    print(f"Validation passed={report['passed']}: {out}")
    if not report["passed"]:
        raise SystemExit("Propagation validation failed; inspect validation.json before generating labels")


def main_generate():
    parser = argparse.ArgumentParser(description=__doc__)
    add_physics_args(parser)
    parser.add_argument("--samples", type=int, default=1000, help="Original apertures; shared across all NF groups")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/forward/v1")
    parser.add_argument("--validation", type=Path, help="Existing matching validation.json; otherwise validate first")
    args = parser.parse_args()
    config, nf_values = config_from_args(args), check_nf(args.nf)
    if args.samples < 24:
        parser.error("--samples must be >=24 for stratified train/val/test splits")
    report = None
    if args.validation:
        report = read_json(args.validation)
        if not report.get("passed") or report.get("config") != config.to_dict() or report.get("nf_values") != list(nf_values):
            parser.error("Validation must pass and match all physics parameters and NF groups")
    out = new_output(args.out)
    if report is None:
        check_dir = out/"validation"
        check_dir.mkdir()
        report = validate(config, nf_values, check_dir)
        if not report["passed"]:
            raise SystemExit("Propagation validation failed; no training data generated")
    write_json(out/"validation.json", report)
    rng = np.random.RandomState(args.seed)
    strategies = [STRATEGIES[i % len(STRATEGIES)] for i in range(args.samples)]
    rng.shuffle(strategies)
    splits = split_ids(strategies, args.seed)
    shape = (args.samples, config.grid, config.grid)
    apertures = np.lib.format.open_memmap(out/"apertures.npy", mode="w+", dtype="float32", shape=shape)
    far_patterns = np.lib.format.open_memmap(out/"fraunhofer_patterns.npy", mode="w+", dtype="float32", shape=shape)
    for i, strategy in enumerate(strategies):
        a = make_aperture(strategy, rng, config.grid)
        apertures[i] = a
        geo = geometry(a, nf_values[0], config)
        far_patterns[i] = postprocess(fraunhofer_raw(a, geo["z_m"], config), config.log_alpha)
    apertures.flush(); far_patterns.flush()
    np.savez(out/"splits.npz", **{k: np.asarray(v, dtype=np.int64) for k, v in splits.items()})
    groups = []
    for nf in nf_values:
        folder = out/group_name(nf)
        folder.mkdir()
        method = "fraunhofer_fft" if nf < config.far_threshold else "scaled_angular_spectrum"
        raw = np.lib.format.open_memmap(folder/"patterns_linear.npy", mode="w+", dtype="float32", shape=shape)
        patterns = np.lib.format.open_memmap(folder/"patterns.npy", mode="w+", dtype="float32", shape=shape)
        metadata = []
        for i in range(args.samples):
            a = np.asarray(apertures[i])
            geo = geometry(a, nf, config)
            intensity = (fraunhofer_raw(a, geo["z_m"], config) if method == "fraunhofer_fft"
                         else angular_spectrum_raw(a, geo["z_m"], config))
            raw[i] = intensity
            patterns[i] = postprocess(intensity, config.log_alpha)
            metadata.append({"id": i, "strategy": strategies[i], "label_method": method, **geo})
            if (i+1) % max(1, args.samples//10) == 0:
                print(f"NF={nf:g} {method}: {i+1}/{args.samples}", flush=True)
        raw.flush(); patterns.flush()
        write_json(folder/"metadata.json", metadata)
        # One example of each family, using the same IDs in every group.
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(8, 3, figsize=(9, 20))
        for row, strategy in enumerate(STRATEGIES):
            i = strategies.index(strategy)
            for col, (value, title) in enumerate(((apertures[i], strategy), (far_patterns[i], "Fraunhofer"), (patterns[i], method))):
                axes[row, col].imshow(value, cmap="gray" if col == 0 else "inferno", vmin=0, vmax=1)
                axes[row, col].set_title(title, fontsize=9); axes[row, col].axis("off")
        fig.suptitle(f"NF={nf:g}; same aperture IDs across groups")
        fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(folder/"preview.png", dpi=100); plt.close(fig)
        groups.append({"name": group_name(nf), "nf": nf, "label_method": method})
    manifest = {"schema": 1, "status": "complete", "config": config.to_dict(), "samples": args.samples,
                "seed": args.seed, "groups": groups, "split_counts": {k: len(v) for k, v in splits.items()},
                "aperture_noise": 0.0, "multiplicative_scaling": False,
                "D_definition": "max occupied axis-aligned pixel span, including gaps; D^2/(lambda*z)",
                "detector_grid": "x_j=(j-N/2)*lambda*z/(N*input_pitch); pitch is sample-dependent and saved",
                "sampling_note": "All methods use the SAME detector coordinates per sample. This is a scaled detector, not a fixed physical camera across z.",
                "geometry_note": "Eight legacy families; slits corrected locally to separate along width. No legacy geometry/output changed.",
                "files": {"apertures.npy": sha256(out/"apertures.npy"), "splits.npz": sha256(out/"splits.npz")}}
    write_json(out/"manifest.json", manifest)
    print(f"Complete: {out}\nSplits: {manifest['split_counts']}")


matplotlib.use("Agg")


class GroupData(Dataset):
    def __init__(self, data_dir, group, split):
        self.root = Path(data_dir)
        self.manifest = read_json(self.root/"manifest.json")
        if self.manifest.get("status") != "complete":
            raise ValueError("Dataset generation is incomplete")
        self.apertures = np.load(self.root/"apertures.npy", mmap_mode="r")
        self.patterns = np.load(self.root/group/"patterns.npy", mmap_mode="r")
        self.far = np.load(self.root/"fraunhofer_patterns.npy", mmap_mode="r")
        self.metadata = read_json(self.root/group/"metadata.json")
        with np.load(self.root/"splits.npz") as splits:
            self.ids = splits[split].copy()
            train_ids = splits["train"].copy()
        expected = (self.manifest["samples"], self.manifest["config"]["grid"], self.manifest["config"]["grid"])
        if self.apertures.shape != expected or self.patterns.shape != expected or self.far.shape != expected:
            raise ValueError("Input/label shapes do not match manifest")
        if any(x.dtype != np.float32 for x in (self.apertures, self.patterns, self.far)):
            raise ValueError("Input and processed labels must be float32")
        if len(self.metadata) != len(self.apertures) or any(m["id"] != i for i, m in enumerate(self.metadata)):
            raise ValueError("Metadata and aperture IDs do not match")
        config = self.manifest["config"]
        expected_group = next(g for g in self.manifest["groups"] if g["name"] == group)
        for m in self.metadata:
            if min(m["z_m"], m["D_m"], m["detector_pitch_m"]) <= 0:
                raise ValueError("Nonpositive physical geometry")
            if not np.isclose(m["D_m"]**2/(config["wavelength_m"]*m["z_m"]), expected_group["nf"], rtol=1e-8):
                raise ValueError("NF/size/distance metadata is inconsistent")
            if not np.isclose(m["detector_pitch_m"], config["wavelength_m"]*m["z_m"]/(config["grid"]*config["input_pitch_m"]), rtol=1e-8, atol=0):
                raise ValueError("Detector coordinates are inconsistent")
            if m["label_method"] != expected_group["label_method"]:
                raise ValueError("Label method metadata is inconsistent")
        self.conditions = np.log(np.array([[m["z_m"], m["detector_pitch_m"]] for m in self.metadata]))
        if not np.isfinite(self.conditions).all():
            raise ValueError("Nonfinite propagation conditions")
        self.condition_mean = self.conditions[train_ids].mean(0)
        self.condition_std = np.maximum(self.conditions[train_ids].std(0), 1e-6)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        i = int(self.ids[index])
        a = torch.from_numpy(np.array(self.apertures[i], copy=True)).unsqueeze(0)
        target = torch.from_numpy(np.array(self.patterns[i], copy=True)).unsqueeze(0)
        condition = torch.tensor((self.conditions[i]-self.condition_mean)/self.condition_std, dtype=torch.float32)
        return a, target, condition, i

    def target_mean(self):
        # Only training labels, not validation or test labels.
        return float(np.mean([self.patterns[int(i)].mean(dtype=np.float64) for i in self.ids]))

    def z_tensor(self, ids, device):
        return torch.tensor([self.metadata[int(i)]["z_m"] for i in ids], dtype=torch.float64, device=device)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA is unavailable in this Python environment")
        return torch.device("cpu")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            value = torch.ones(1, 1, 16, 16, device="cuda", requires_grad=True)
            layer = torch.nn.Conv2d(1, 1, 3, padding=1).cuda()
            (torch.fft.fft2(value).abs().mean()+layer(value).mean()).backward()
            torch.cuda.synchronize()
        return torch.device("cuda")
    except RuntimeError as error:
        if requested == "cuda":
            raise RuntimeError("CUDA is detected but FFT/conv/backward kernels failed. Use a compatible PyTorch environment or --device cpu.") from error
        print("CUDA kernel check failed; auto mode is using CPU. Select a compatible PyTorch environment for GPU training.", flush=True)
        return torch.device("cpu")


def aggregate_rows(rows):
    return {key: float(np.mean([r[key] for r in rows])) for key in ("mae", "rmse", "psnr_db", "corr")}


@torch.no_grad()
def score(model, loader, device, physics_only=False):
    model.eval()
    rows = []
    for a, target, condition, ids in loader:
        a, condition = a.to(device), condition.to(device)
        z = loader.dataset.z_tensor(ids, device)
        predicted = model(a, z) if physics_only else model(a, condition, z)[0]
        for ident, p, t in zip(ids.tolist(), predicted[:, 0].cpu().numpy(), target[:, 0].numpy()):
            rows.append({"id": ident, **image_metrics(p, t)})
    return aggregate_rows(rows), rows


def epoch_pass(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    totals = {k: 0.0 for k in ("total", "l1", "freq", "ssim", "res")}
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for a, target, condition, ids in loader:
            a, target, condition = a.to(device), target.to(device), condition.to(device)
            z = loader.dataset.z_tensor(ids, device)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            pred, _, delta = model(a, condition, z)
            loss, stats = criterion(pred, target, delta)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            if optimizer:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            totals["total"] += loss.item()*len(a)
            for key in stats:
                totals[key] += stats[key]*len(a)
            count += len(a)
    return {k: v/count for k, v in totals.items()}


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def check_dataset(root, manifest):
    if not read_json(root/"validation.json").get("passed"):
        raise ValueError("Dataset has no passed propagation validation")
    for filename, expected in manifest["files"].items():
        if sha256(root/filename) != expected:
            raise ValueError(f"Dataset file changed: {filename}")
    with np.load(root/"splits.npz") as split:
        ids = [set(split[k].tolist()) for k in ("train", "val", "test")]
        if any(not x for x in ids) or any(ids[i] & ids[j] for i in range(3) for j in range(i+1, 3)):
            raise ValueError("Empty or overlapping train/val/test split")
        if set.union(*ids) != set(range(manifest["samples"])):
            raise ValueError("Split does not cover the original aperture IDs")


def run_group(args, group, output_root, device):
    group_started = time.perf_counter()
    name, nf = group["name"], group["nf"]
    sets = {k: GroupData(args.data, name, k) for k in ("train", "val", "test")}
    loaders = {k: DataLoader(v, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers) for k, v in sets.items()}
    config = PhysicsConfig(**sets["train"].manifest["config"])
    propagator = ("fraunhofer" if nf < config.far_threshold else "asm") if args.propagator == "hybrid" else args.propagator
    physics = (AngularSpectrumDetectorLayer(config, args.asm_precision) if propagator == "asm"
               else FraunhoferDetectorLayer(config.log_alpha)).to(device)
    physics_label = "ASM" if propagator == "asm" else "Fraunhofer"
    # A strict implementation check is separate from expected approximation error.
    a, target, cond, ident = next(iter(loaders["val"]))
    z = sets["val"].z_tensor(ident, device)
    with torch.no_grad():
        simulated = physics(a.to(device), z).cpu().numpy()[:, 0]
    if propagator == "asm":
        # Numerical implementation check, independent from target agreement.
        count = min(2, len(a))
        cached = np.stack([postprocess(angular_spectrum_raw(a[j, 0].numpy(),
                    sets["val"].metadata[int(ident[j])]["z_m"], config), config.log_alpha) for j in range(count)])
        simulated = simulated[:count]
    else:
        cached = np.asarray(sets["val"].far[ident.numpy()])
    implementation_error = float(np.mean(np.abs(simulated-cached)))
    if implementation_error > 1e-5:
        raise ValueError(f"Torch/Numpy {physics_label} implementation mismatch: {implementation_error:g}")
    with StageTimer(device) as pure_val_timer:
        val_physics, _ = score(physics, loaders["val"], device, True)
    print(f"NF={nf:g} pure {physics_label} validation: {val_physics}; implementation MAE={implementation_error:.3e}", flush=True)
    if args.dry_run:
        for kind in args.models:
            seed_all(args.seed)
            model = ForwardModel(kind, args.base_ch, physics.alpha, sets["train"].target_mean(),
                                 propagator, config, args.asm_precision).to(device)
            with torch.no_grad():
                pred, _, delta = model(a.to(device), cond.to(device), z)
            if pred.shape != target.shape or not torch.isfinite(pred).all():
                raise ValueError("Model forward check failed")
            if kind == "physics" and torch.count_nonzero(delta).item():
                raise ValueError("Residual is not zero at initialization")
            print(f"  {kind}: forward OK, parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
        return []
    out = output_root/name
    out.mkdir()
    pure_stem = "pure_asm" if propagator == "asm" else "pure_physics"
    write_json(out/(pure_stem+"_val.json"), val_physics)
    with StageTimer(device) as pure_test_timer:
        summary, pure_rows = score(physics, loaders["test"], device, True)
    write_csv(out/(pure_stem+"_test.csv"), pure_rows)
    pure_name = "angular_spectrum_only" if propagator == "asm" else "fraunhofer_only"
    result_rows = [{"nf": nf, "model": pure_name, "propagator": propagator, **summary}]
    time_rows = [{"nf": nf, "model": pure_name, "propagator": propagator, "epochs": 0,
                  "train_seconds": 0.0, "validation_seconds": pure_val_timer.seconds,
                  "fit_wall_seconds": 0.0, "test_seconds": pure_test_timer.seconds,
                  "model_total_seconds": pure_val_timer.seconds+pure_test_timer.seconds}]
    for kind in args.models:
        model_started = time.perf_counter()
        model_started_utc = utc_now()
        folder = out/kind
        folder.mkdir()
        seed_all(args.seed)
        model = ForwardModel(kind, args.base_ch, physics.alpha, sets["train"].target_mean(),
                             propagator, config, args.asm_precision).to(device)
        criterion = Loss(lambda_freq=args.lambda_freq, lambda_ssim=args.lambda_ssim,
                         lambda_res=args.lambda_res if kind == "physics" else 0.0)
        if criterion.lambda_ssim != args.lambda_ssim:
            raise RuntimeError("Requested SSIM unavailable; install its dependency or explicitly use --lambda-ssim 0")
        train_loader = DataLoader(sets["train"], batch_size=args.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(args.seed), num_workers=args.workers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        with StageTimer(device) as initial_timer:
            initial, _ = score(model, loaders["val"], device)
        write_json(folder/"initial_val.json", initial)
        def save_checkpoint(epoch, validation_l1):
            torch.save({"state_dict": model.state_dict(), "kind": kind, "base_ch": args.base_ch,
                        "nf": nf, "epoch": epoch, "val_l1": validation_l1,
                        "condition_mean": sets["train"].condition_mean.tolist(),
                        "condition_std": sets["train"].condition_std.tolist(),
                        "physics_config": sets["train"].manifest["config"], "training_args": vars(args),
                        "propagator": propagator, "asm_precision": args.asm_precision},
                       folder/"best_model.pt")
        # Zero residual may already be optimal in the far-field groups. Keep
        # epoch 0 eligible rather than forcing selection of a worse update.
        history, best = [], initial["mae"]
        save_checkpoint(0, best)
        synchronize(device)
        setup_seconds = time.perf_counter()-model_started
        fit_started = time.perf_counter()
        train_seconds = val_seconds = 0.0
        for epoch in range(args.epochs):
            epoch_started = time.perf_counter()
            with StageTimer(device) as train_timer:
                train_stats = epoch_pass(model, train_loader, criterion, device, optimizer)
            with StageTimer(device) as val_timer:
                val_stats = epoch_pass(model, loaders["val"], criterion, device)
            train_seconds += train_timer.seconds
            val_seconds += val_timer.seconds
            row = {"epoch": epoch+1, "lr": optimizer.param_groups[0]["lr"],
                   **{"train_"+k: v for k, v in train_stats.items()},
                   **{"val_"+k: v for k, v in val_stats.items()}}
            checkpoint_seconds = 0.0
            if val_stats["l1"] < best:
                best = val_stats["l1"]
                checkpoint_started = time.perf_counter()
                save_checkpoint(epoch+1, best)
                checkpoint_seconds = time.perf_counter()-checkpoint_started
            scheduler.step()
            row.update(train_seconds=train_timer.seconds, validation_seconds=val_timer.seconds,
                       checkpoint_seconds=checkpoint_seconds, epoch_seconds=time.perf_counter()-epoch_started,
                       elapsed_fit_seconds=time.perf_counter()-fit_started)
            history.append(row)
            write_csv(folder/"history.csv", history)
            write_json(folder/"timing.json", {"status": "running", "started_at_utc": model_started_utc,
                       "epochs_completed": epoch+1, "train_seconds": train_seconds,
                       "validation_seconds": val_seconds, "fit_wall_seconds": time.perf_counter()-fit_started})
            print(f"NF={nf:g} {physics_label} {kind} {epoch+1}/{args.epochs}: train L1={train_stats['l1']:.3e}, val L1={val_stats['l1']:.3e} | train {train_timer.seconds:.2f}s, val {val_timer.seconds:.2f}s, epoch {row['epoch_seconds']:.2f}s | elapsed {duration(row['elapsed_fit_seconds'])}", flush=True)
        synchronize(device)
        fit_wall_seconds = time.perf_counter()-fit_started
        checkpoint = torch.load(folder/"best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        write_json(folder/"selection.json", {"best_epoch": checkpoint["epoch"],
                                             "best_validation_l1": checkpoint["val_l1"],
                                             "initial_validation_mae": initial["mae"]})
        with StageTimer(device) as test_timer:
            summary, sample_rows = score(model, loaders["test"], device)
        write_json(folder/"test_metrics.json", summary)
        write_csv(folder/"test_per_sample.csv", sample_rows)
        result_rows.append({"nf": nf, "model": kind, "propagator": propagator, **summary})
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy([r["epoch"] for r in history], [max(r["train_l1"], 1e-12) for r in history], label="train")
        ax.semilogy([r["epoch"] for r in history], [max(r["val_l1"], 1e-12) for r in history], label="val")
        ax.axhline(max(initial["mae"], 1e-12), ls=":", color="gray", label="initial val (epoch 0)")
        ax.set(xlabel="Epoch", ylabel="Spatial L1", title=f"NF={nf:g}, {kind}")
        ax.legend(); fig.tight_layout(); fig.savefig(folder/"loss_curve.png", dpi=120); plt.close(fig)
        model.eval()
        with torch.no_grad():
            pred, sim, delta = model(a.to(device), cond.to(device), z)
        count = min(4, len(a))
        fig, axes = plt.subplots(count, 5, figsize=(15, 3*count), squeeze=False)
        for r in range(count):
            far = physics(a[r:r+1].to(device), z[r:r+1])[0, 0].detach().cpu().numpy()
            prediction = pred[r, 0].cpu().numpy()
            values = (a[r, 0].numpy(), far, prediction, target[r, 0].numpy(), np.abs(prediction-target[r, 0].numpy()))
            for c, (value, title) in enumerate(zip(values, ("Aperture", physics_label, kind, "Label", "Absolute error"))):
                im = axes[r, c].imshow(value, cmap="gray" if c == 0 else "inferno", vmin=0, vmax=None if c == 4 else 1)
                axes[r, c].set_title(title); axes[r, c].axis("off")
                if c == 4: fig.colorbar(im, ax=axes[r, c], fraction=0.046)
        fig.tight_layout(); fig.savefig(folder/"val_samples.png", dpi=110); plt.close(fig)
        synchronize(device)
        model_total_seconds = time.perf_counter()-model_started
        timing_row = {"nf": nf, "model": kind, "propagator": propagator, "epochs": args.epochs,
                      "train_seconds": train_seconds, "validation_seconds": val_seconds,
                      "fit_wall_seconds": fit_wall_seconds, "test_seconds": test_timer.seconds,
                      "model_total_seconds": model_total_seconds}
        time_rows.append(timing_row)
        write_json(folder/"timing.json", {**timing_row, "status": "completed",
                   "started_at_utc": model_started_utc, "finished_at_utc": utc_now(),
                   "setup_seconds": setup_seconds, "initial_validation_seconds": initial_timer.seconds,
                   "mean_train_epoch_seconds": train_seconds/args.epochs,
                   "mean_validation_epoch_seconds": val_seconds/args.epochs,
                   "scope": "CUDA-synchronized wall time; training/validation include loader and transfers; fit wall includes epoch I/O; model total additionally includes setup, selection, testing and plots."})
        print(f"NF={nf:g} {kind} finished: fit {duration(fit_wall_seconds)}, model total {duration(model_total_seconds)}", flush=True)
    group_seconds = time.perf_counter()-group_started
    write_json(out/"group_timing.json", {"nf": nf, "propagator": propagator, "group_seconds": group_seconds,
               "pure_validation_seconds": pure_val_timer.seconds, "pure_test_seconds": pure_test_timer.seconds,
               "rows": time_rows})
    return result_rows


def main_train():
    run_started = time.perf_counter()
    run_started_utc = utc_now()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/forward/v1")
    parser.add_argument("--out", default=None, help="New directory only; defaults to a timestamped runs/forward folder")
    parser.add_argument("--groups", type=float, nargs="+", help="NF values; default all groups")
    parser.add_argument("--models", nargs="+", choices=("physics", "baseline"), default=("physics", "baseline"))
    parser.add_argument("--propagator", choices=("fraunhofer", "asm", "hybrid"), default="fraunhofer",
                        help="hybrid keeps FFT below the dataset threshold and ASM above it")
    parser.add_argument("--asm-precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-freq", type=float, default=0.1)
    parser.add_argument("--lambda-ssim", type=float, default=0.0, help="Same coefficient for both trainable models")
    parser.add_argument("--lambda-res", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0, help="Windows-safe default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Read-only data/model forward checks; does not train or create run outputs")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.base_ch < 8 or args.base_ch % 8 or args.cpu_threads < 1:
        parser.error("epochs/batch size must be positive; base-ch must be a multiple of 8")
    if args.lr <= 0 or min(args.lambda_freq, args.lambda_ssim, args.lambda_res) < 0:
        parser.error("Learning rate must be positive and loss coefficients nonnegative")
    root = Path(args.data).resolve()
    manifest = read_json(root/"manifest.json")
    check_dataset(root, manifest)
    groups = manifest["groups"]
    if args.groups:
        if any(not any(abs(g["nf"]-x) < 1e-10 for g in groups) for x in args.groups):
            parser.error("Requested NF group is absent from this dataset")
        groups = [g for g in groups if any(abs(g["nf"]-x) < 1e-10 for x in args.groups)]
    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    print(f"Device: {device}; independently training {len(groups)} NF groups", flush=True)
    out = None if args.dry_run else new_output(args.out or ("runs/forward/"+datetime.now().strftime("%Y%m%d_%H%M%S")))
    if out:
        write_json(out/"run_config.json", {"arguments": vars(args), "dataset_manifest_sha256": sha256(root/"manifest.json"),
                                          "selection": "best validation spatial L1; test used only after checkpoint selection"})
    rows, time_rows, group_times = [], [], {}
    for group in groups:
        rows.extend(run_group(args, group, out, device))
        if out:
            write_csv(out/"summary.csv", rows)
            group_timing = read_json(out/group["name"]/"group_timing.json")
            time_rows.extend(group_timing["rows"])
            group_times[group["name"]] = group_timing["group_seconds"]
            write_csv(out/"timing_summary.csv", time_rows)
            write_json(out/"timing.json", {"status": "running", "started_at_utc": run_started_utc,
                       "wall_seconds": time.perf_counter()-run_started, "group_seconds": group_times})
    if out:
        synchronize(device)
        wall_seconds = time.perf_counter()-run_started
        write_json(out/"timing.json", {"status": "completed", "started_at_utc": run_started_utc,
                   "finished_at_utc": utc_now(), "wall_seconds": wall_seconds, "group_seconds": group_times,
                   "scope": "main entry through all groups; includes data checks and device probe, excludes Python import/startup and final timing-file write"})
        print(f"Total run wall time: {duration(wall_seconds)}", flush=True)
    print("Dry-run passed; no training performed" if args.dry_run else f"Training completed: {out}")


COMMANDS = {"validate": main_validate, "generate": main_generate, "train": main_train}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: python -m aperture.forward {validate|generate|train} [options]")
        raise SystemExit(2)
    COMMANDS[sys.argv.pop(1)]()


if __name__ == "__main__":
    main()
