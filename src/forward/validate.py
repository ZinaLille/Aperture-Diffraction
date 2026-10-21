"""Validate propagation BEFORE generating training labels. No training here."""
import argparse
import csv
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.common.physics import (PhysicsConfig, DEFAULT_NF, check_nf, geometry, postprocess,
                                fraunhofer_raw, angular_spectrum_raw,
                                rayleigh_sommerfeld_points, image_metrics)
from src.common.io_utils import new_output, write_json


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


def main():
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


if __name__ == "__main__":
    main()
