"""Generate new NF-group data without touching data/ or legacy results."""
import argparse
from pathlib import Path
import numpy as np
from .apertures import STRATEGIES, make_aperture, split_ids
from .physics import (check_nf, group_name, geometry, postprocess,
                      fraunhofer_raw, angular_spectrum_raw)
from .io_utils import new_output, write_json, read_json, sha256
from .validate import add_physics_args, config_from_args, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_physics_args(parser)
    parser.add_argument("--samples", type=int, default=1000, help="Original apertures; shared across all NF groups")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="fresnel_data/v1")
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


if __name__ == "__main__":
    main()
