"""Generate shared apertures and the seven Fresnel-number label groups.

Usage:
    python -m aperture.generate --out data/forward/v1 --samples 1000

Propagation is validated internally before any training array is written; the
report is stored as ``validation.json`` inside the output directory.
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
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aperture.apertures import STRATEGIES, make_aperture, split_ids
from aperture.physics import (PhysicsConfig, DEFAULT_NF, check_nf, group_name, geometry, postprocess,
                      fraunhofer_raw, angular_spectrum_raw,
                      rayleigh_sommerfeld_points, image_metrics)
from aperture.utils import new_output, write_json, sha256

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




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_physics_args(parser)
    parser.add_argument("--samples", type=int, default=1000, help="Original apertures; shared across all NF groups")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/forward/v1")
    args = parser.parse_args()
    config, nf_values = config_from_args(args), check_nf(args.nf)
    if args.samples < 24:
        parser.error("--samples must be >=24 for stratified train/val/test splits")
    out = new_output(args.out)
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





if __name__ == "__main__":
    main()
