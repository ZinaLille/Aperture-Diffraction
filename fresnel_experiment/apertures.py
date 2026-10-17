"""Reuse the eight legacy geometry families without input noise or scaling."""
import numpy as np
import generate_data as legacy

STRATEGIES = ("multi_elements", "slits", "grating", "hole_array", "letter",
              "annular", "mixed_grating", "polygons")


def make_aperture(strategy, rng, grid=256):
    # Legacy geometry ranges are defined on 256 pixels. Resize its binary
    # pixel cells only if a smaller diagnostic grid is explicitly requested.
    a = np.zeros((256, 256), dtype=np.float32)
    if strategy == "slits":
        # Separate parallel slits along their WIDTH axis. The legacy helper
        # offsets along their length and merges them; leave that script intact.
        n = rng.randint(1, 6)
        width, gap, height = rng.randint(3, 12), rng.randint(15, 35), rng.randint(80, 150)
        angle = rng.uniform(0, np.pi)
        for j in range(n):
            offset = (j-(n-1)/2)*(width+gap)
            legacy.draw_rect(a, 128+offset*np.cos(angle), 128+offset*np.sin(angle), width, height, angle)
    else:
        getattr(legacy, "strategy_"+strategy)(a, rng, 256)
    if grid != 256:
        ids = np.minimum(((np.arange(grid)+0.5)*256/grid).astype(int), 255)
        a = a[np.ix_(ids, ids)].copy()
    if not np.any(a):
        raise ValueError("Aperture disappeared on the requested grid; use a finer grid")
    return a


def split_ids(strategies, seed=42):
    """Stratified, shared original-aperture split: approximately 80/10/10."""
    rng = np.random.RandomState(seed)
    result = {"train": [], "val": [], "test": []}
    for name in STRATEGIES:
        ids = np.flatnonzero(np.asarray(strategies) == name)
        rng.shuffle(ids)
        if len(ids) < 3:
            raise ValueError("Need at least 3 original apertures per strategy (24 total)")
        holdout = max(1, int(round(len(ids)*0.1)))
        result["test"].extend(ids[:holdout].tolist())
        result["val"].extend(ids[holdout:2*holdout].tolist())
        result["train"].extend(ids[2*holdout:].tolist())
    return {k: sorted(v) for k, v in result.items()}
