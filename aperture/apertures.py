"""Eight aperture families, shared stratified splits and geometry helpers."""
import numpy as np

def draw_rect(a, cx, cy, w, h, angle=0):
    """旋转矩形"""
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    x = xx - cx
    y = yy - cy
    ca, sa = np.cos(-angle), np.sin(-angle)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    mask = (np.abs(xr) <= w / 2) & (np.abs(yr) <= h / 2)
    a[mask] = 1.0


def draw_circle(a, cx, cy, r):
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
    a[mask] = 1.0


def draw_ellipse(a, cx, cy, rx, ry, angle=0):
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    x = xx - cx
    y = yy - cy
    ca, sa = np.cos(-angle), np.sin(-angle)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    mask = (xr / rx)**2 + (yr / ry)**2 <= 1
    a[mask] = 1.0


def draw_ring(a, cx, cy, r_in, r_out):
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    d2 = (xx - cx)**2 + (yy - cy)**2
    mask = (d2 >= r_in**2) & (d2 <= r_out**2)
    a[mask] = 1.0


def draw_polygon(a, cx, cy, n_sides, radius, angle=0):
    """正多边形"""
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + angle
    verts = np.stack([cx + radius * np.cos(angles),
                      cy + radius * np.sin(angles)], axis=1)
    _fill_polygon(a, verts)


def draw_irregular_polygon(a, cx, cy, n_sides, r_min, r_max, angle=0,
                            rng=None):
    """不规则多边形: 每个顶点距离中心随机"""
    if rng is None:
        rng = np.random
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + angle
    angles = angles + rng.uniform(-0.3, 0.3, n_sides)
    radii = rng.uniform(r_min, r_max, n_sides)
    verts = np.stack([cx + radii * np.cos(angles),
                      cy + radii * np.sin(angles)], axis=1)
    _fill_polygon(a, verts)


def draw_star(a, cx, cy, n_points, r_outer, r_inner, angle=0):
    """星形"""
    angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False) + angle
    radii = np.tile([r_outer, r_inner], n_points)
    verts = np.stack([cx + radii * np.cos(angles),
                      cy + radii * np.sin(angles)], axis=1)
    _fill_polygon(a, verts)


def draw_cross(a, cx, cy, arm_length, arm_width, angle=0):
    """十字形"""
    draw_rect(a, cx, cy, arm_length, arm_width, angle)
    draw_rect(a, cx, cy, arm_width, arm_length, angle)


def _fill_polygon(a, verts):
    """射线法填充多边形"""
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    inside = np.zeros_like(xx, dtype=bool)
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        cond = ((y1 > yy) != (y2 > yy))
        with np.errstate(divide='ignore', invalid='ignore'):
            x_int = (x2 - x1) * (yy - y1) / (y2 - y1 + 1e-12) + x1
        inside ^= cond & (xx < x_int)
    a[inside] = 1.0


def draw_grating(a, cx, cy, w, h, period, duty, angle):
    """有限尺寸光栅"""
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    x = xx - cx
    y = yy - cy
    ca, sa = np.cos(-angle), np.sin(-angle)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    in_rect = (np.abs(xr) <= w / 2) & (np.abs(yr) <= h / 2)
    phase = (xr % period) / period
    bars = phase < duty
    a[in_rect & bars] = 1.0


def draw_multi_slits(a, cx, cy, n_slits, slit_width, gap, height, angle):
    """多缝"""
    total = n_slits * slit_width + (n_slits - 1) * gap
    start = -total / 2 + slit_width / 2
    for i in range(n_slits):
        offset = start + i * (slit_width + gap)
        dx = offset * np.cos(angle + np.pi / 2)
        dy = offset * np.sin(angle + np.pi / 2)
        draw_rect(a, cx + dx, cy + dy, slit_width, height, angle)


# ============================================================
# 8 种光阑生成策略
# ============================================================
def strategy_multi_elements(a, rng, size):
    """策略1: 多个不同元素随机放置"""
    n = rng.randint(2, 5)
    placed = []
    
    for _ in range(n):
        for _attempt in range(15):
            cx = rng.randint(40, size - 40)
            cy = rng.randint(40, size - 40)
            ok = True
            for (px, py, pr) in placed:
                if (cx - px)**2 + (cy - py)**2 < (pr + 30)**2:
                    ok = False
                    break
            if ok:
                break
        
        elem = rng.choice(['rect', 'circle', 'ellipse', 'polygon'])
        if elem == 'rect':
            w, h = rng.randint(15, 60), rng.randint(15, 60)
            ang = rng.uniform(0, np.pi)
            draw_rect(a, cx, cy, w, h, ang)
            placed.append((cx, cy, max(w, h)))
        elif elem == 'circle':
            r = rng.randint(10, 35)
            draw_circle(a, cx, cy, r)
            placed.append((cx, cy, r))
        elif elem == 'ellipse':
            rx, ry = rng.randint(10, 45), rng.randint(10, 45)
            ang = rng.uniform(0, np.pi)
            draw_ellipse(a, cx, cy, rx, ry, ang)
            placed.append((cx, cy, max(rx, ry)))
        else:
            n_sides = rng.randint(3, 8)
            r = rng.randint(15, 40)
            ang = rng.uniform(0, 2 * np.pi)
            draw_polygon(a, cx, cy, n_sides, r, ang)
            placed.append((cx, cy, r))


def strategy_slits(a, rng, size):
    """策略2: 多缝"""
    n = rng.choice([1, 2, 3, 4, 5])
    slit_width = rng.randint(3, 12)
    gap = rng.randint(15, 60)
    height = rng.randint(80, 180)
    angle = rng.uniform(0, np.pi)
    cx = size / 2 + rng.randint(-30, 30)
    cy = size / 2 + rng.randint(-30, 30)
    draw_multi_slits(a, cx, cy, n, slit_width, gap, height, angle)


def strategy_grating(a, rng, size):
    """策略3: 有限尺寸光栅"""
    w = rng.randint(80, 200)
    h = rng.randint(80, 200)
    period = rng.randint(8, 30)
    duty = rng.uniform(0.3, 0.7)
    angle = rng.uniform(0, np.pi)
    cx = size / 2 + rng.randint(-40, 40)
    cy = size / 2 + rng.randint(-40, 40)
    draw_grating(a, cx, cy, w, h, period, duty, angle)


def strategy_hole_array(a, rng, size):
    """策略4: 孔洞阵列"""
    pattern = rng.choice(['grid', 'ring', 'random'])
    
    if pattern == 'grid':
        n_x = rng.randint(3, 8)
        n_y = rng.randint(3, 8)
        r = rng.randint(3, 10)
        margin = 40
        xs = np.linspace(margin, size - margin, n_x)
        ys = np.linspace(margin, size - margin, n_y)
        for x in xs:
            for y in ys:
                draw_circle(a, x, y, r)
    
    elif pattern == 'ring':
        n = rng.randint(6, 16)
        r_hole = rng.randint(3, 8)
        r_ring = rng.randint(40, 90)
        cx, cy = size / 2, size / 2
        for k in range(n):
            ang = 2 * np.pi * k / n + rng.uniform(-0.1, 0.1)
            x = cx + r_ring * np.cos(ang)
            y = cy + r_ring * np.sin(ang)
            draw_circle(a, x, y, r_hole)
    
    else:
        n = rng.randint(5, 20)
        for _ in range(n):
            cx = rng.randint(30, size - 30)
            cy = rng.randint(30, size - 30)
            r = rng.randint(3, 12)
            draw_circle(a, cx, cy, r)


def strategy_letter(a, rng, size):
    """策略5: 字母形状"""
    letter = rng.choice(['T', 'L', 'H', 'X', 'O', 'C', 'F', 'E'])
    thickness = rng.randint(8, 18)
    scale = rng.randint(60, 100)
    cx, cy = size / 2, size / 2
    
    if letter == 'T':
        draw_rect(a, cx, cy - scale / 2, scale, thickness, 0)
        draw_rect(a, cx, cy, thickness, scale, 0)
    elif letter == 'L':
        draw_rect(a, cx - scale / 2, cy, thickness, scale, 0)
        draw_rect(a, cx, cy + scale / 2, scale, thickness, 0)
    elif letter == 'H':
        draw_rect(a, cx - scale / 2, cy, thickness, scale, 0)
        draw_rect(a, cx + scale / 2, cy, thickness, scale, 0)
        draw_rect(a, cx, cy, scale, thickness, 0)
    elif letter == 'X':
        draw_rect(a, cx, cy, thickness, scale, np.pi / 4)
        draw_rect(a, cx, cy, thickness, scale, -np.pi / 4)
    elif letter == 'O':
        draw_ring(a, cx, cy, scale / 2 - thickness, scale / 2)
    elif letter == 'C':
        yy, xx = np.mgrid[:size, :size]
        d2 = (xx - cx)**2 + (yy - cy)**2
        outer = d2 <= (scale / 2)**2
        inner = d2 <= (scale / 2 - thickness)**2
        angle = np.arctan2(yy - cy, xx - cx)
        gap = angle > -np.pi / 4
        a[outer & ~inner & ~gap] = 1.0
    elif letter == 'F':
        draw_rect(a, cx - scale / 2, cy, thickness, scale, 0)
        draw_rect(a, cx - scale / 4, cy - scale / 2, scale / 2, thickness, 0)
        draw_rect(a, cx - scale / 4, cy, scale / 3, thickness, 0)
    elif letter == 'E':
        draw_rect(a, cx - scale / 2, cy, thickness, scale, 0)
        draw_rect(a, cx - scale / 4, cy - scale / 2, scale / 2, thickness, 0)
        draw_rect(a, cx - scale / 4, cy, scale / 3, thickness, 0)
        draw_rect(a, cx - scale / 4, cy + scale / 2, scale / 2, thickness, 0)


def strategy_annular(a, rng, size):
    """策略6: 环形/扇形"""
    pattern = rng.choice(['single_ring', 'multi_ring', 'sector', 'annular_array'])
    
    if pattern == 'single_ring':
        cx = size / 2 + rng.randint(-30, 30)
        cy = size / 2 + rng.randint(-30, 30)
        r_out = rng.randint(30, 80)
        r_in = r_out - rng.randint(5, 20)
        draw_ring(a, cx, cy, r_in, r_out)
    
    elif pattern == 'multi_ring':
        cx, cy = size / 2, size / 2
        n = rng.randint(2, 5)
        for k in range(n):
            r_out = rng.randint(20, 100)
            r_in = r_out - rng.randint(3, 8)
            draw_ring(a, cx, cy, r_in, r_out)
    
    elif pattern == 'sector':
        yy, xx = np.mgrid[:size, :size]
        cx, cy = size / 2, size / 2
        d2 = (xx - cx)**2 + (yy - cy)**2
        angle = np.arctan2(yy - cy, xx - cx)
        r_max = rng.randint(40, 100)
        n_sectors = rng.choice([2, 3, 4])
        sector_width = 2 * np.pi / n_sectors
        phase = rng.uniform(0, 2 * np.pi)
        for k in range(n_sectors):
            ang_center = phase + k * sector_width
            ang_mask = np.abs(np.angle(np.exp(1j * (angle - ang_center)))) < sector_width / 2.5
            a[(d2 <= r_max**2) & ang_mask] = 1.0
    
    else:
        cx, cy = size / 2, size / 2
        n_rings = rng.randint(2, 4)
        for k in range(1, n_rings + 1):
            r_out = 30 * k
            r_in = r_out - rng.randint(3, 8)
            if r_out < size // 2 - 10:
                draw_ring(a, cx, cy, r_in, r_out)


def strategy_mixed_grating(a, rng, size):
    """策略7: 光栅 + 缺陷"""
    w = rng.randint(100, 220)
    h = rng.randint(100, 220)
    period = rng.randint(8, 25)
    duty = rng.uniform(0.3, 0.7)
    angle = rng.uniform(0, np.pi)
    cx = size / 2 + rng.randint(-20, 20)
    cy = size / 2 + rng.randint(-20, 20)
    
    draw_grating(a, cx, cy, w, h, period, duty, angle)
    
    n_holes = rng.randint(1, 5)
    for _ in range(n_holes):
        hx = rng.randint(0, size)
        hy = rng.randint(0, size)
        hr = rng.randint(3, 15)
        yy, xx = np.mgrid[:size, :size]
        a[(xx - hx)**2 + (yy - hy)**2 <= hr**2] = 0.0
    
    if rng.rand() > 0.5:
        cx2 = rng.randint(30, size - 30)
        cy2 = rng.randint(30, size - 30)
        r2 = rng.randint(15, 40)
        draw_circle(a, cx2, cy2, r2)


def strategy_polygons(a, rng, size):
    """策略8: 各种多边形"""
    kind = rng.choice(['regular', 'irregular', 'star', 'cross', 'triangle_mix'])
    cx = size / 2 + rng.randint(-40, 40)
    cy = size / 2 + rng.randint(-40, 40)
    
    if kind == 'regular':
        n = rng.randint(3, 9)
        r = rng.randint(40, 90)
        ang = rng.uniform(0, 2 * np.pi)
        draw_polygon(a, cx, cy, n, r, ang)
    
    elif kind == 'irregular':
        n = rng.randint(3, 8)
        r_min, r_max = 30, 90
        ang = rng.uniform(0, 2 * np.pi)
        draw_irregular_polygon(a, cx, cy, n, r_min, r_max, ang, rng)
    
    elif kind == 'star':
        n = rng.randint(3, 8)
        r_out = rng.randint(50, 100)
        r_in = r_out * rng.uniform(0.3, 0.6)
        ang = rng.uniform(0, 2 * np.pi)
        draw_star(a, cx, cy, n, r_out, r_in, ang)
    
    elif kind == 'cross':
        arm = rng.randint(60, 120)
        width = rng.randint(8, 25)
        ang = rng.uniform(0, np.pi)
        draw_cross(a, cx, cy, arm, width, ang)
    
    else:
        n = rng.randint(2, 5)
        for _ in range(n):
            tcx = rng.randint(40, size - 40)
            tcy = rng.randint(40, size - 40)
            r = rng.randint(20, 50)
            ang = rng.uniform(0, 2 * np.pi)
            draw_polygon(a, tcx, tcy, 3, r, ang)


STRATEGIES = ("multi_elements", "slits", "grating", "hole_array", "letter",
              "annular", "mixed_grating", "polygons")


STRATEGY_FUNCS = {
    "multi_elements": strategy_multi_elements,
    "slits": strategy_slits,
    "grating": strategy_grating,
    "hole_array": strategy_hole_array,
    "letter": strategy_letter,
    "annular": strategy_annular,
    "mixed_grating": strategy_mixed_grating,
    "polygons": strategy_polygons,
}


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
            draw_rect(a, 128+offset*np.cos(angle), 128+offset*np.sin(angle), width, height, angle)
    else:
        STRATEGY_FUNCS[strategy](a, rng, 256)
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
