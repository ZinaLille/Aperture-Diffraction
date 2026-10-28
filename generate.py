"""阶段一数据生成：光阑 -> 菲涅耳数 -> 衍射图样。

七组菲涅耳数共用同一批原始光阑。NF 小于切换点时用夫琅禾费 FFT 生成标签，
其余用缩放角谱；光阑按 8 类几何策略生成，并做分层 80/10/10 划分。

用法：
    python generate.py --samples 1000 --out data/v1
    python generate.py --grid 64 --samples 24 --out data/demo
    python generate.py --nf 0.01 0.03 0.1 0.3 1 3 5 --skip-validation

输出（输出目录必须不存在）：
    apertures.npy            (N, grid, grid) float32，各组共享
    splits.npz               train / val / test 的原始光阑编号
    validation.json          传播验证报告
    propagation_check.png    FFT 与角谱对照
    manifest.json            最后写，status=complete 才能训练
    nf_*/patterns.npy        该 NF 的 log 压缩标签
    nf_*/preview.png         八类光阑示例
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_NF = (0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 5.0)
STRATEGIES = ("multi_elements", "slits", "grating", "hole_array", "letter",
              "annular", "mixed_grating", "polygons")


# ============================================================
# 1. 传播参数与物理约定
# ============================================================
@dataclass(frozen=True)
class PhysicsConfig:
    grid: int = 256
    input_pitch_m: float = 10e-6
    wavelength_m: float = 532e-9
    far_threshold: float = 0.1
    oversample: int = 2
    pad_factor: int = 2
    log_alpha: float = 100.0

    def __post_init__(self):
        if self.grid < 32 or self.grid % 8:
            raise ValueError("grid 必须 >=32 且能被 8 整除")
        if any(not np.isfinite(x) or x <= 0
               for x in (self.input_pitch_m, self.wavelength_m, self.log_alpha)):
            raise ValueError("像素间距、波长和 log_alpha 必须为正")
        if self.oversample < 1 or self.pad_factor < 1:
            raise ValueError("oversample 和 pad_factor 必须 >=1")
        if not 0 < self.far_threshold <= 5:
            raise ValueError("far_threshold 必须在 (0,5] 内")

    def to_dict(self):
        return asdict(self)


def check_nf(values):
    values = tuple(float(x) for x in values)
    if not values or any(not np.isfinite(x) or not 0 < x <= 5 for x in values):
        raise ValueError("菲涅耳数必须有限且位于 (0,5]")
    if len(set(values)) != len(values):
        raise ValueError("菲涅耳数不能重复")
    if len(set(format(x, ".6g") for x in values)) != len(values):
        raise ValueError("菲涅耳数过近，组目录名会冲突")
    return values


def group_name(nf):
    return "nf_" + format(float(nf), ".6g").replace(".", "p")


def support_width(aperture, pitch):
    """D：非零像素包围盒的较长轴向跨度，包含孔间距。"""
    rows, cols = np.nonzero(aperture > 0)
    if not len(rows):
        raise ValueError("空光阑没有定义的菲涅耳数")
    return max(np.ptp(rows) + 1, np.ptp(cols) + 1) * pitch


def geometry(aperture, nf, config):
    """由光阑跨度 D 得到 z = D^2/(lambda*NF) 与观察面像素间距。"""
    d = support_width(aperture, config.input_pitch_m)
    z = d * d / (config.wavelength_m * nf)
    pitch = config.wavelength_m * z / (config.grid * config.input_pitch_m)
    return {"D_m": d, "z_m": z, "nf": float(nf),
            "detector_pitch_m": pitch,
            "detector_fov_m": config.grid * pitch}


def postprocess(raw, alpha=100.0):
    """峰值归一化 + log1p 压缩，输出 [0,1]。"""
    raw = np.maximum(np.asarray(raw, dtype=np.float64), 0)
    peak = raw.max()
    if not np.isfinite(peak):
        raise ValueError("传播强度包含非有限值")
    if peak == 0:
        return np.zeros_like(raw, dtype=np.float32)
    value = np.log1p(alpha * (raw / peak))
    return (value / value.max()).astype(np.float32)


def fraunhofer_raw(aperture, z, config):
    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture)))
    return np.abs(field * config.input_pitch_m**2 / (config.wavelength_m * z))**2


def _fine_aperture(aperture, oversample, pad_factor):
    # 光阑像素是常透过率单元；偶数倍细分需要半个细像素的原点修正。
    fine = np.repeat(np.repeat(aperture, oversample, 0), oversample, 1)
    m = len(fine) * pad_factor
    result = np.zeros((m, m), dtype=np.float64)
    offset = (m - len(fine)) // 2
    result[offset:offset + len(fine), offset:offset + len(fine)] = fine
    return result, (oversample - 1) / 2


def angular_spectrum_raw(aperture, z, config):
    """缩放角谱：保留精确角谱色散，在夫琅禾费观察网格上取样。"""
    c, pad = config.oversample, config.pad_factor
    fine, origin_shift = _fine_aperture(aperture, c, pad)
    m = len(fine)
    dx = config.input_pitch_m / c
    frequency = np.fft.fftfreq(m, dx)
    q = config.wavelength_m**2 * (frequency[:, None]**2 + frequency[None, :]**2)
    propagating = q < 1
    root = np.sqrt(np.maximum(1 - q, 0))
    # sqrt(1-q)-1+q/2 的稳定写法，避免大 z 下的相消误差。
    residual_phase = -(2 * np.pi / config.wavelength_m) * z * q * q / (2 * (1 + root)**2)
    correction = np.exp(1j * residual_phase) * propagating
    spectrum = np.fft.fft2(np.fft.ifftshift(fine))
    corrected = np.fft.fftshift(np.fft.ifft2(spectrum * correction))
    coords = (np.arange(m) - m // 2 - origin_shift) * dx
    chirp = np.exp(1j * np.pi * coords**2 / (config.wavelength_m * z))
    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
        corrected * chirp[:, None] * chirp[None, :])))
    # 输出 x_j=(j-N/2)*lambda*z/(N*input_pitch)；补零只是更密地采样同一网格。
    indices = m // 2 + (np.arange(config.grid) - config.grid // 2) * pad
    field = field[np.ix_(indices, indices)]
    return np.abs(field * dx**2 / (config.wavelength_m * z))**2


def rayleigh_sommerfeld_points(aperture, z, points_xy, config, oversample=4):
    """独立的 RS1 中点积分，只用于少量抽样验证。"""
    fine = np.repeat(np.repeat(aperture, oversample, 0), oversample, 1)
    dx = config.input_pitch_m / oversample
    origin = (oversample - 1) / 2
    coords = (np.arange(len(fine)) - len(fine) // 2 - origin) * dx
    iy, ix = np.nonzero(fine)
    xx, yy, weights = coords[ix], coords[iy], fine[iy, ix]
    k = 2 * np.pi / config.wavelength_m
    values = []
    for x, y in points_xy:
        rho2 = (x - xx)**2 + (y - yy)**2
        r = np.sqrt(z * z + rho2)
        phase = k * rho2 / (r + z)          # r-z 的稳定形式
        kernel = z / (1j * config.wavelength_m * r * r) * (1 + 1j / (k * r)) * np.exp(1j * phase)
        values.append(abs(np.sum(weights * kernel) * dx * dx)**2)
    return np.asarray(values)


def image_metrics(pred, target):
    difference = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    mse = float(np.mean(difference**2))
    rmse = float(np.sqrt(mse))
    corr = 0.0
    if min(np.std(pred), np.std(target)) > 1e-12:
        corr = float(np.corrcoef(np.ravel(pred), np.ravel(target))[0, 1])
    return {"mae": float(np.mean(np.abs(difference))), "rmse": rmse,
            "psnr_db": float(20 * np.log10(1 / max(rmse, 1e-12))), "corr": corr}


# ============================================================
# 2. 八类光阑（几何与 legacy 保持一致）
# ============================================================
def draw_rect(a, cx, cy, w, h, angle=0):
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    x, y = xx - cx, yy - cy
    ca, sa = np.cos(-angle), np.sin(-angle)
    xr, yr = ca * x - sa * y, sa * x + ca * y
    a[(np.abs(xr) <= w / 2) & (np.abs(yr) <= h / 2)] = 1.0


def draw_circle(a, cx, cy, r):
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    a[(xx - cx)**2 + (yy - cy)**2 <= r**2] = 1.0


def draw_ellipse(a, cx, cy, rx, ry, angle=0):
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    x, y = xx - cx, yy - cy
    ca, sa = np.cos(-angle), np.sin(-angle)
    xr, yr = ca * x - sa * y, sa * x + ca * y
    a[(xr / rx)**2 + (yr / ry)**2 <= 1] = 1.0


def draw_ring(a, cx, cy, r_in, r_out):
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    d2 = (xx - cx)**2 + (yy - cy)**2
    a[(d2 >= r_in**2) & (d2 <= r_out**2)] = 1.0


def _fill_polygon(a, verts):
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    inside = np.zeros_like(xx, dtype=bool)
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        cond = (y1 > yy) != (y2 > yy)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = (x2 - x1) * (yy - y1) / (y2 - y1 + 1e-12) + x1
        inside ^= cond & (xx < x_int)
    a[inside] = 1.0


def draw_polygon(a, cx, cy, n_sides, radius, angle=0):
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + angle
    verts = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
    _fill_polygon(a, verts)


def draw_irregular_polygon(a, cx, cy, n_sides, r_min, r_max, angle, rng):
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + angle
    angles = angles + rng.uniform(-0.3, 0.3, n_sides)
    radii = rng.uniform(r_min, r_max, n_sides)
    _fill_polygon(a, np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1))


def draw_star(a, cx, cy, n_points, r_outer, r_inner, angle=0):
    angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False) + angle
    radii = np.tile([r_outer, r_inner], n_points)
    _fill_polygon(a, np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1))


def draw_cross(a, cx, cy, arm_length, arm_width, angle=0):
    draw_rect(a, cx, cy, arm_length, arm_width, angle)
    draw_rect(a, cx, cy, arm_width, arm_length, angle)


def draw_grating(a, cx, cy, w, h, period, duty, angle):
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    x, y = xx - cx, yy - cy
    ca, sa = np.cos(-angle), np.sin(-angle)
    xr, yr = ca * x - sa * y, sa * x + ca * y
    in_rect = (np.abs(xr) <= w / 2) & (np.abs(yr) <= h / 2)
    a[in_rect & ((xr % period) / period < duty)] = 1.0


def strategy_multi_elements(a, rng, size):
    placed = []
    for _ in range(rng.randint(2, 5)):
        for _ in range(15):
            cx, cy = rng.randint(40, size - 40), rng.randint(40, size - 40)
            if all((cx - px)**2 + (cy - py)**2 >= (pr + 30)**2 for px, py, pr in placed):
                break
        elem = rng.choice(["rect", "circle", "ellipse", "polygon"])
        if elem == "rect":
            w, h, ang = rng.randint(15, 60), rng.randint(15, 60), rng.uniform(0, np.pi)
            draw_rect(a, cx, cy, w, h, ang)
            placed.append((cx, cy, max(w, h)))
        elif elem == "circle":
            r = rng.randint(10, 35)
            draw_circle(a, cx, cy, r)
            placed.append((cx, cy, r))
        elif elem == "ellipse":
            rx, ry, ang = rng.randint(10, 45), rng.randint(10, 45), rng.uniform(0, np.pi)
            draw_ellipse(a, cx, cy, rx, ry, ang)
            placed.append((cx, cy, max(rx, ry)))
        else:
            r, ang = rng.randint(15, 40), rng.uniform(0, 2 * np.pi)
            draw_polygon(a, cx, cy, rng.randint(3, 8), r, ang)
            placed.append((cx, cy, r))


def strategy_slits(a, rng, size):
    n = rng.randint(1, 6)
    width, gap, height = rng.randint(3, 12), rng.randint(15, 35), rng.randint(80, 150)
    angle = rng.uniform(0, np.pi)
    for j in range(n):
        offset = (j - (n - 1) / 2) * (width + gap)
        draw_rect(a, 128 + offset * np.cos(angle), 128 + offset * np.sin(angle),
                  width, height, angle)


def strategy_grating(a, rng, size):
    draw_grating(a, size / 2 + rng.randint(-40, 40), size / 2 + rng.randint(-40, 40),
                 rng.randint(80, 200), rng.randint(80, 200),
                 rng.randint(8, 30), rng.uniform(0.3, 0.7), rng.uniform(0, np.pi))


def strategy_hole_array(a, rng, size):
    pattern = rng.choice(["grid", "ring", "random"])
    if pattern == "grid":
        xs = np.linspace(40, size - 40, rng.randint(3, 8))
        ys = np.linspace(40, size - 40, rng.randint(3, 8))
        r = rng.randint(3, 10)
        for x in xs:
            for y in ys:
                draw_circle(a, x, y, r)
    elif pattern == "ring":
        n, r_hole, r_ring = rng.randint(6, 16), rng.randint(3, 8), rng.randint(40, 90)
        for k in range(n):
            ang = 2 * np.pi * k / n + rng.uniform(-0.1, 0.1)
            draw_circle(a, size / 2 + r_ring * np.cos(ang),
                        size / 2 + r_ring * np.sin(ang), r_hole)
    else:
        for _ in range(rng.randint(5, 20)):
            draw_circle(a, rng.randint(30, size - 30), rng.randint(30, size - 30),
                        rng.randint(3, 12))


def strategy_letter(a, rng, size):
    letter = rng.choice(["T", "L", "H", "X", "O", "C", "F", "E"])
    t, s = rng.randint(8, 18), rng.randint(60, 100)
    cx, cy = size / 2, size / 2
    if letter == "T":
        draw_rect(a, cx, cy - s / 2, s, t)
        draw_rect(a, cx, cy, t, s)
    elif letter == "L":
        draw_rect(a, cx - s / 2, cy, t, s)
        draw_rect(a, cx, cy + s / 2, s, t)
    elif letter == "H":
        draw_rect(a, cx - s / 2, cy, t, s)
        draw_rect(a, cx + s / 2, cy, t, s)
        draw_rect(a, cx, cy, s, t)
    elif letter == "X":
        draw_rect(a, cx, cy, t, s, np.pi / 4)
        draw_rect(a, cx, cy, t, s, -np.pi / 4)
    elif letter == "O":
        draw_ring(a, cx, cy, s / 2 - t, s / 2)
    elif letter == "C":
        yy, xx = np.mgrid[:size, :size]
        d2 = (xx - cx)**2 + (yy - cy)**2
        outer = d2 <= (s / 2)**2
        inner = d2 <= (s / 2 - t)**2
        gap = np.arctan2(yy - cy, xx - cx) > -np.pi / 4
        a[outer & ~inner & ~gap] = 1.0
    elif letter == "F":
        draw_rect(a, cx - s / 2, cy, t, s)
        draw_rect(a, cx - s / 4, cy - s / 2, s / 2, t)
        draw_rect(a, cx - s / 4, cy, s / 3, t)
    else:
        draw_rect(a, cx - s / 2, cy, t, s)
        draw_rect(a, cx - s / 4, cy - s / 2, s / 2, t)
        draw_rect(a, cx - s / 4, cy, s / 3, t)
        draw_rect(a, cx - s / 4, cy + s / 2, s / 2, t)


def strategy_annular(a, rng, size):
    pattern = rng.choice(["single_ring", "multi_ring", "sector", "annular_array"])
    cx, cy = size / 2, size / 2
    if pattern == "single_ring":
        r_out = rng.randint(30, 80)
        draw_ring(a, cx + rng.randint(-30, 30), cy + rng.randint(-30, 30),
                  r_out - rng.randint(5, 20), r_out)
    elif pattern == "multi_ring":
        for _ in range(rng.randint(2, 5)):
            r_out = rng.randint(20, 100)
            draw_ring(a, cx, cy, r_out - rng.randint(3, 8), r_out)
    elif pattern == "sector":
        yy, xx = np.mgrid[:size, :size]
        d2 = (xx - cx)**2 + (yy - cy)**2
        angle = np.arctan2(yy - cy, xx - cx)
        r_max, n = rng.randint(40, 100), rng.choice([2, 3, 4])
        phase, width = rng.uniform(0, 2 * np.pi), 2 * np.pi / n
        for k in range(n):
            delta = np.abs(np.angle(np.exp(1j * (angle - phase - k * width))))
            a[(d2 <= r_max**2) & (delta < width / 2.5)] = 1.0
    else:
        for k in range(1, rng.randint(2, 4) + 1):
            r_out = 30 * k
            if r_out < size // 2 - 10:
                draw_ring(a, cx, cy, r_out - rng.randint(3, 8), r_out)


def strategy_mixed_grating(a, rng, size):
    draw_grating(a, size / 2 + rng.randint(-20, 20), size / 2 + rng.randint(-20, 20),
                 rng.randint(100, 220), rng.randint(100, 220),
                 rng.randint(8, 25), rng.uniform(0.3, 0.7), rng.uniform(0, np.pi))
    for _ in range(rng.randint(1, 5)):
        hx, hy, hr = rng.randint(0, size), rng.randint(0, size), rng.randint(3, 15)
        yy, xx = np.mgrid[:size, :size]
        a[(xx - hx)**2 + (yy - hy)**2 <= hr**2] = 0.0
    if rng.rand() > 0.5:
        draw_circle(a, rng.randint(30, size - 30), rng.randint(30, size - 30),
                    rng.randint(15, 40))


def strategy_polygons(a, rng, size):
    kind = rng.choice(["regular", "irregular", "star", "cross", "triangle_mix"])
    cx, cy = size / 2 + rng.randint(-40, 40), size / 2 + rng.randint(-40, 40)
    if kind == "regular":
        draw_polygon(a, cx, cy, rng.randint(3, 9), rng.randint(40, 90),
                     rng.uniform(0, 2 * np.pi))
    elif kind == "irregular":
        draw_irregular_polygon(a, cx, cy, rng.randint(3, 8), 30, 90,
                               rng.uniform(0, 2 * np.pi), rng)
    elif kind == "star":
        r_out = rng.randint(50, 100)
        draw_star(a, cx, cy, rng.randint(3, 8), r_out,
                  r_out * rng.uniform(0.3, 0.6), rng.uniform(0, 2 * np.pi))
    elif kind == "cross":
        draw_cross(a, cx, cy, rng.randint(60, 120), rng.randint(8, 25),
                   rng.uniform(0, np.pi))
    else:
        for _ in range(rng.randint(2, 5)):
            draw_polygon(a, rng.randint(40, size - 40), rng.randint(40, size - 40),
                         3, rng.randint(20, 50), rng.uniform(0, 2 * np.pi))


STRATEGY_FUNCS = {
    "multi_elements": strategy_multi_elements, "slits": strategy_slits,
    "grating": strategy_grating, "hole_array": strategy_hole_array,
    "letter": strategy_letter, "annular": strategy_annular,
    "mixed_grating": strategy_mixed_grating, "polygons": strategy_polygons,
}


def strategy_assignment(samples, seed=42):
    """复现 generate() 内部的光阑策略分配（只依赖 samples 与 seed）。

    生成时先 round-robin 再 shuffle；shuffle 是 RandomState(seed) 的第一次使用，
    因此可以从 manifest 的 samples/seed 还原每个光阑属于哪一类，无需改数据格式。
    """
    strategies = [STRATEGIES[i % len(STRATEGIES)] for i in range(samples)]
    np.random.RandomState(seed).shuffle(strategies)
    return strategies


def make_aperture(strategy, rng, grid=256):
    a = np.zeros((256, 256), dtype=np.float32)
    STRATEGY_FUNCS[strategy](a, rng, 256)
    if grid != 256:
        ids = np.minimum(((np.arange(grid) + 0.5) * 256 / grid).astype(int), 255)
        a = a[np.ix_(ids, ids)].copy()
    if not np.any(a):
        raise ValueError("光阑在目标网格上消失，请使用更细的 grid")
    return a


def split_ids(strategies, seed=42):
    """按策略分层、七组共享的 80/10/10 划分。"""
    rng = np.random.RandomState(seed)
    result = {"train": [], "val": [], "test": []}
    for name in STRATEGIES:
        ids = np.flatnonzero(np.asarray(strategies) == name)
        rng.shuffle(ids)
        if len(ids) < 3:
            raise ValueError("每类至少需要 3 个原始光阑（共 24 个）")
        holdout = max(1, int(round(len(ids) * 0.1)))
        result["test"].extend(ids[:holdout].tolist())
        result["val"].extend(ids[holdout:2 * holdout].tolist())
        result["train"].extend(ids[2 * holdout:].tolist())
    return {k: sorted(v) for k, v in result.items()}


# ============================================================
# 3. 生成前传播验证
# ============================================================
def standard_apertures(n):
    yy, xx = np.indices((n, n))
    circle = ((xx - n // 2)**2 + (yy - n // 2)**2 <= (n / 8)**2).astype(np.float32)
    width = max(2, n // 32)
    slit = np.zeros((n, n), np.float32)
    slit[n // 4:3 * n // 4, n // 2 - width // 2:n // 2 + width // 2] = 1
    double = np.zeros_like(slit)
    for center in (n // 2 - n // 10, n // 2 + n // 10):
        double[n // 4:3 * n // 4, center - width // 2:center + width // 2] = 1
    grating = np.zeros_like(slit)
    for start in range(n // 4, 3 * n // 4, max(4, n // 16)):
        grating[n // 4:3 * n // 4, start:start + max(2, n // 32)] = 1
    return {"circle": circle, "single_slit": slit, "double_slit": double, "grating": grating}


def validate(config, nf_values, out):
    """在写入训练数组前检查传播近似、网格收敛与独立 RS1 抽样。"""
    standards, rows, checks = standard_apertures(config.grid), [], []
    fig, axes = plt.subplots(len(standards), len(nf_values),
                             figsize=(2.6 * len(nf_values), 2.6 * len(standards)), squeeze=False)
    for row, (name, a) in enumerate(standards.items()):
        for col, nf in enumerate(nf_values):
            geo = geometry(a, nf, config)
            far = postprocess(fraunhofer_raw(a, geo["z_m"], config), config.log_alpha)
            ref = postprocess(angular_spectrum_raw(a, geo["z_m"], config), config.log_alpha)
            metric = image_metrics(far, ref)
            rows.append({"shape": name, "nf": nf, **geo, **metric})
            axes[row, col].imshow(ref, cmap="inferno", vmin=0, vmax=1)
            axes[row, col].set_title(f"{name}, NF={nf:g}\nFFT/ASM MAE={metric['mae']:.1e}",
                                     fontsize=8)
            axes[row, col].axis("off")
            if nf < config.far_threshold:
                checks.append({"check": "far_agreement", "shape": name, "nf": nf,
                               "value": metric["mae"], "limit": 0.005,
                               "passed": metric["mae"] < 0.005})
        print(f"  验证 {name}", flush=True)
    fig.tight_layout()
    fig.savefig(out / "propagation_check.png", dpi=120)
    plt.close(fig)

    # 网格收敛 + 独立 RS1：避免把单一离散化当成真值。
    circle = standards["circle"]
    for nf in (0.1, 1.0, 5.0):
        geo = geometry(circle, nf, config)
        raw = angular_spectrum_raw(circle, geo["z_m"], config)
        base = postprocess(raw, config.log_alpha)
        for label, other in (("padding", replace(config, pad_factor=config.pad_factor * 2)),
                             ("refinement", replace(config, oversample=config.oversample * 2))):
            refined = postprocess(angular_spectrum_raw(circle, geo["z_m"], other), config.log_alpha)
            value = image_metrics(base, refined)["mae"]
            checks.append({"check": label, "shape": "circle", "nf": nf,
                           "value": value, "limit": 0.002, "passed": value < 0.002})
        ids = np.array([[config.grid // 2, config.grid // 2],
                        [config.grid // 2, config.grid // 2 + 1],
                        [config.grid // 2, config.grid // 2 + 3],
                        [config.grid // 2 + 3, config.grid // 2 + 3]])
        points = (ids[:, ::-1] - config.grid // 2) * geo["detector_pitch_m"]
        direct = rayleigh_sommerfeld_points(circle, geo["z_m"], points, config,
                                            oversample=config.oversample * 2)
        value = float(np.max(np.abs(direct - raw[ids[:, 0], ids[:, 1]])) /
                      max(float(direct.max()), 1e-30))
        checks.append({"check": "independent_RS1", "shape": "circle", "nf": nf,
                       "value": value, "limit": 0.02, "passed": value < 0.02})
        print(f"  收敛/RS1 NF={nf:g}", flush=True)

    # 切换点两侧都用同一观察坐标报告差异，不隐藏方法切换。
    switch = []
    for name, a in standards.items():
        geo = geometry(a, config.far_threshold, config)
        value = image_metrics(
            postprocess(fraunhofer_raw(a, geo["z_m"], config), config.log_alpha),
            postprocess(angular_spectrum_raw(a, geo["z_m"], config), config.log_alpha))["mae"]
        switch.append({"shape": name, "nf": config.far_threshold, "mae": value})
        checks.append({"check": "switch_agreement", "shape": name, "nf": config.far_threshold,
                       "value": value, "limit": 0.005, "passed": value < 0.005})

    with (out / "propagation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"schema": 2, "config": config.to_dict(), "nf_values": list(nf_values),
              "reference": "缩放角谱（含精确角谱色散修正）",
              "checks": checks, "switch": switch,
              "passed": all(c["passed"] for c in checks),
              "note": "这些是本次实验的数值验收条件，不是普适物理边界。"}
    return report


# ============================================================
# 4. 生成七组数据
# ============================================================
def new_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"输出目录已存在：{path}；请换一个新名字")
    path.mkdir(parents=True)
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def add_physics_args(parser):
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--pitch-um", type=float, default=10.0)
    parser.add_argument("--wavelength-nm", type=float, default=532.0)
    parser.add_argument("--far-threshold", type=float, default=0.1)
    parser.add_argument("--oversample", type=int, default=2)
    parser.add_argument("--pad-factor", type=int, default=2)
    parser.add_argument("--nf", type=float, nargs="+", default=DEFAULT_NF)


def config_from_args(args):
    return PhysicsConfig(grid=args.grid, input_pitch_m=args.pitch_um * 1e-6,
                         wavelength_m=args.wavelength_nm * 1e-9,
                         far_threshold=args.far_threshold,
                         oversample=args.oversample, pad_factor=args.pad_factor)


def generate(config, nf_values, samples, seed, out):
    rng = np.random.RandomState(seed)
    strategies = [STRATEGIES[i % len(STRATEGIES)] for i in range(samples)]
    rng.shuffle(strategies)
    splits = split_ids(strategies, seed)
    shape = (samples, config.grid, config.grid)

    print(f"生成 {samples} 个光阑 ({config.grid}x{config.grid}) ...", flush=True)
    apertures = np.lib.format.open_memmap(out / "apertures.npy", mode="w+",
                                          dtype="float32", shape=shape)
    for i, strategy in enumerate(strategies):
        apertures[i] = make_aperture(strategy, rng, config.grid)
        if (i + 1) % max(1, samples // 10) == 0:
            print(f"  光阑 {i + 1}/{samples}", flush=True)
    apertures.flush()
    np.savez(out / "splits.npz",
             **{k: np.asarray(v, dtype=np.int64) for k, v in splits.items()})

    groups = []
    for nf in nf_values:
        folder = out / group_name(nf)
        folder.mkdir()
        method = "fraunhofer_fft" if nf < config.far_threshold else "scaled_angular_spectrum"
        patterns = np.lib.format.open_memmap(folder / "patterns.npy", mode="w+",
                                            dtype="float32", shape=shape)
        for i in range(samples):
            a = np.asarray(apertures[i])
            z = geometry(a, nf, config)["z_m"]
            raw = (fraunhofer_raw(a, z, config) if method == "fraunhofer_fft"
                   else angular_spectrum_raw(a, z, config))
            patterns[i] = postprocess(raw, config.log_alpha)
            if (i + 1) % max(1, samples // 10) == 0:
                print(f"  NF={nf:g} {method} {i + 1}/{samples}", flush=True)
        patterns.flush()
        _preview(config, apertures, patterns, strategies, nf, folder)
        groups.append({"name": group_name(nf), "nf": nf, "label_method": method})
        print(f"NF={nf:g} 完成：{folder}", flush=True)

    manifest = {"schema": 2, "status": "complete", "config": config.to_dict(),
                "samples": samples, "seed": seed, "groups": groups,
                "split_counts": {k: len(v) for k, v in splits.items()},
                "D_definition": "非零像素包围盒较长跨度（含孔间距）；z=D^2/(lambda*NF)",
                "detector_grid": "x_j=(j-N/2)*lambda*z/(N*input_pitch)，随样本变化",
                "note": "七组共享同一批光阑与同一 train/val/test 划分。"}
    write_json(out / "manifest.json", manifest)
    return manifest


def _preview(config, apertures, patterns, strategies, nf, folder):
    """八类光阑各取一个：光阑 | 同坐标夫琅禾费 | 本组 NF 标签。"""
    fig, axes = plt.subplots(len(STRATEGIES), 3, figsize=(9, 2.3 * len(STRATEGIES)))
    for row, strategy in enumerate(STRATEGIES):
        i = strategies.index(strategy)
        a = np.asarray(apertures[i])
        far = postprocess(fraunhofer_raw(a, geometry(a, nf, config)["z_m"], config),
                          config.log_alpha)
        for col, (value, title) in enumerate(((a, strategy), (far, "Fraunhofer"),
                                              (patterns[i], f"NF={nf:g} label"))):
            axes[row, col].imshow(value, cmap="gray" if col == 0 else "inferno", vmin=0, vmax=1)
            axes[row, col].set_title(title, fontsize=8)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(folder / "preview.png", dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_physics_args(parser)
    parser.add_argument("--samples", type=int, default=1000,
                        help="原始光阑数，七组共用")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/v1")
    parser.add_argument("--skip-validation", action="store_true",
                        help="跳过生成前的传播验证（不建议用于正式数据）")
    args = parser.parse_args()
    if args.samples < 24:
        parser.error("--samples 至少为 24（每类 3 个光阑）")
    config = config_from_args(args)
    nf_values = check_nf(args.nf)
    out = new_output(args.out)

    if not args.skip_validation:
        print("传播验证 ...", flush=True)
        report = validate(config, nf_values, out)
        write_json(out / "validation.json", report)
        if not report["passed"]:
            raise SystemExit("传播验证未通过：请检查 validation.json，未写入训练数据")
        print(f"验证通过：{sum(c['passed'] for c in report['checks'])}/{len(report['checks'])} 项", flush=True)
    else:
        print("已跳过传播验证", flush=True)
        write_json(out / "validation.json", {"skipped": True})

    manifest = generate(config, nf_values, args.samples, args.seed, out)
    print(f"完成：{out}")
    print(f"划分：{manifest['split_counts']}")


if __name__ == "__main__":
    main()
