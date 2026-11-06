"""阶段二 v2 数据生成：光阑 + 随机相位毛玻璃 -> 含扰动的衍射图样（GT）。

与 `generate.py` 的关系：

- 光阑的八类几何、分层划分、`postprocess` 的 log 标签全部复用 `generate.py`；
  用**同一个 seed 和 samples** 生成时，`apertures.npy` 与 v1 逐位相同，
  因此 v1 / v2 可以直接按样本号对照，唯一的差别就是有没有毛玻璃。
- 新增一块**随机相位薄屏**（毛玻璃），紧贴光阑放在物面上（thin phase screen）。
  真值是

      I_glass(x) = | P_z[ e^{iφ(x')} · A(x') ](x) |²

  先峰值归一化再 log1p 压缩，和 v1 的标签是同一套后处理。把相位屏放在物面上
  是标准的薄散射体模型，物理上就是「毛玻璃紧贴光阑背面」；离光阑几个 mm 的间距
  在这个尺度下对图样的影响只有 ~1e-5（远场半径 λz/pitch 是米量级），所以没有必要
  再引入一个中间面（引入后观察面网格的定义也会变）。

物理约定（与 `generate.py` 完全一致，`scaled_field` 是它的复数版本）：

- 观察面网格 `x_j = (j - N/2) · λ·z / (N·Δx)`，`z = D²/(λ·NF)`，
  `D` 是非零像素包围盒的较长跨度。
- NF < far_threshold 用夫琅禾费 FFT，其余用缩放角谱；两条路径都作用在玻璃后的
  **复振幅**上，所以毛玻璃在两种近似下被一致地计入。

为什么要存玻璃相位：

- 训练/评估只换数据集就够了：输出目录与 `generate.py` 完全同构，`train.py` / `evaluate.py`
  一行不改就能读（网络只看得到光阑和无玻璃物理层，毛玻璃的影响由残差 U-Net 去学）。
- `glass_phase.npy` 只在需要**复核**反设计时用：`design.py` 的正向是训练好的网络（不需要 φ），
  但它的 `--check-glass` 需要一个真实相位屏把设计放回解析模型复算一遍。真实实验里没有 φ 时，
  这一步换成用实测图样对比即可。

用法：

    python generate_v2.py --samples 1000 --glass-shared --out data/v2_shared   # 一块固定玻璃
    python generate_v2.py --samples 1000 --out data/v2                         # 逐样本随机（对照）
    python generate_v2.py --samples 1000 --glass-preset strong --out data/v2_strong
    python generate_v2.py --grid 64 --samples 32 --out data/demo_v2
    python generate_v2.py --glass-correlation-um 500 --glass-rms-rad 0.8 --out data/v2_custom

输出（目录必须不存在；manifest 最后写，status=complete 才能训练）：

    apertures.npy           (N, grid, grid) float32，与 v1 同 seed 时逐位相同
    glass_phase.npy         (N, grid, grid) float32 弧度；--glass-shared 时是 (grid, grid)
    splits.npz              train / val / test 的原始光阑编号
    validation.json         相位屏与两种标签路径的数值验收报告
    glass_metrics.csv       每个标准形状的玻璃扰动强度
    glass_check.png         光阑 | 相位屏 | 无玻璃 | 含玻璃 | 残差
    manifest.json           config / glass / groups / split_counts
    nf_*/patterns.npy       该 NF 的含玻璃标签
    nf_*/patterns_clean.npy 无玻璃理想图样（仅 --save-clean）
    nf_*/preview.png        八类光阑示例
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate import (STRATEGIES, PhysicsConfig, add_physics_args, check_nf,
                      config_from_args, fraunhofer_raw, geometry, group_name,
                      image_metrics, make_aperture, new_output, postprocess,
                      split_ids, standard_apertures, write_json)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 预设按「相关长度 / 掩模尺寸」给，保证 64 和 256 网格下是同一个定性区间；
# 想要物理尺寸就直接用 --glass-correlation-um 覆盖。
GLASS_PRESETS = {
    "weak": {"correlation_ratio": 0.31, "rms_rad": 0.6},
    "moderate": {"correlation_ratio": 0.12, "rms_rad": 1.2},
    "strong": {"correlation_ratio": 0.06, "rms_rad": 2.3},
    "groundglass": {"correlation_ratio": 0.016, "rms_rad": 3.0},
}
GLASS_SEED_OFFSET = 1000003   # 与光阑 RNG 分开，保证光阑和 v1 逐位一致


# ============================================================
# 1. 毛玻璃：随机相位屏
# ============================================================
@dataclass(frozen=True)
class GlassConfig:
    """薄相位屏毛玻璃；相位是各向同性高斯随机场，相关长度按 1/e 自相关半径定义。"""

    correlation_m: float = 300e-6
    rms_rad: float = 1.2
    shared: bool = False

    def __post_init__(self):
        if not np.isfinite(self.correlation_m) or self.correlation_m <= 0:
            raise ValueError("毛玻璃相关长度必须为正")
        if not np.isfinite(self.rms_rad) or self.rms_rad < 0:
            raise ValueError("相位起伏 rms 必须非负")

    def to_dict(self):
        return asdict(self)


def phase_screen(rng, grid, pitch, correlation_m, rms_rad):
    """高斯随机相位屏（弧度）。

    白噪声经高斯低通：滤波核 std = correlation/2，于是相位自相关的 1/e 半径
    正好等于 correlation_m，`measured_correlation` 里量的就是它。
    """
    noise = rng.standard_normal((grid, grid))
    frequency = np.fft.fftfreq(grid, d=pitch)
    sigma = 0.5 * correlation_m
    kernel = np.exp(-2 * np.pi**2 * sigma**2
                    * (frequency[:, None]**2 + frequency[None, :]**2))
    phase = np.real(np.fft.ifft2(np.fft.fft2(noise) * kernel))
    phase = phase - phase.mean()
    scale = phase.std()
    if scale < 1e-15:
        raise ValueError("相位屏退化（相关长度过大或网格过小），请减小相关长度")
    return (phase * (rms_rad / scale)).astype(np.float32)


def measured_correlation(phases, pitch):
    """从自相关函数的 1/e 半径反推实际相关长度（沿过原点的一行取径向剖面）。"""
    phases = np.atleast_3d(np.asarray(phases, dtype=np.float64))
    values = []
    for phase in phases:
        spectrum = np.fft.fft2(phase - phase.mean())
        auto = np.fft.fftshift(np.real(np.fft.ifft2(spectrum * np.conj(spectrum))))
        auto /= max(float(auto.max()), 1e-30)
        centre = auto.shape[0] // 2
        profile = auto[centre, centre:]
        radii = np.arange(len(profile)) * pitch
        below = np.flatnonzero(profile <= np.exp(-1.0))
        values.append(float(radii[below[0]]) if len(below) else float(radii[-1]))
    return float(np.mean(values))


# ============================================================
# 2. 复振幅传播（与 generate.angular_spectrum_raw 逐行一致）
# ============================================================
def scaled_field(field, z, config, oversample=None, pad_factor=None):
    """`generate.angular_spectrum_raw` 的复数版本：支持复输入、保留相位。

    返回去掉观察面二次相位的复振幅，强度与 angular_spectrum_raw 相同。
    """
    c = config.oversample if oversample is None else oversample
    pad = config.pad_factor if pad_factor is None else pad_factor
    fine = np.repeat(np.repeat(np.asarray(field, dtype=np.complex128), c, 0), c, 1)
    m = len(fine) * pad
    offset = (m - len(fine)) // 2
    buffer = np.zeros((m, m), dtype=np.complex128)
    buffer[offset:offset + len(fine), offset:offset + len(fine)] = fine
    dx = config.input_pitch_m / c
    frequency = np.fft.fftfreq(m, d=dx)
    q = config.wavelength_m**2 * (frequency[:, None]**2 + frequency[None, :]**2)
    root = np.sqrt(np.maximum(1 - q, 0))
    residual_phase = -(2 * np.pi / config.wavelength_m) * z * q * q / (2 * (1 + root)**2)
    corrected = np.fft.fftshift(np.fft.ifft2(
        np.fft.fft2(np.fft.ifftshift(buffer)) * np.exp(1j * residual_phase) * (q < 1)))
    coords = (np.arange(m) - m // 2 - (c - 1) / 2) * dx
    chirp = np.exp(1j * np.pi * coords**2 / (config.wavelength_m * z))
    out = np.fft.fftshift(np.fft.fft2(
        np.fft.ifftshift(corrected * chirp[:, None] * chirp[None, :])))
    indices = m // 2 + (np.arange(config.grid) - config.grid // 2) * pad
    return out[np.ix_(indices, indices)] * dx**2 / (config.wavelength_m * z)


def field_after_glass(aperture, phase):
    """物面上的复振幅 A·e^{iφ}。"""
    return (np.asarray(aperture, dtype=np.float64)
            * np.exp(1j * np.asarray(phase, dtype=np.float64))).astype(np.complex128)


def raw_intensity(field, z, config, method):
    """玻璃后的复振幅 -> 观察面强度（与 v1 两种标签路径一致）。"""
    if method == "fraunhofer_fft":
        return fraunhofer_raw(field, z, config)
    return np.abs(scaled_field(field, z, config))**2


# ============================================================
# 3. 生成前的数值验收
# ============================================================
def _shapes(config):
    shapes = standard_apertures(config.grid)
    n = config.grid
    letter = np.zeros((n, n), np.float32)
    bar = max(2, n // 32)
    letter[n // 3:2 * n // 3, n // 2 - n // 16:n // 2 - n // 16 + bar] = 1
    letter[n // 3:n // 3 + bar, n // 2 - n // 16:n // 2 + n // 6] = 1
    letter[n // 2:n // 2 + bar, n // 2 - n // 16:n // 2 + n // 10] = 1
    shapes["letter_F"] = letter
    return shapes


def validate(config, glass, nf_values, out):
    """写训练数组之前：相位屏统计、两条标签路径一致性、零相位恒等。"""
    rng = np.random.RandomState(20261106)
    checks, info = [], {}
    shapes = _shapes(config)
    n, pitch = config.grid, config.input_pitch_m

    # (1) 相位屏统计是否符合请求（rms 与相关长度）
    screens = np.stack([phase_screen(rng, n, pitch, glass.correlation_m, glass.rms_rad)
                        for _ in range(8)])
    measured_rms = float(np.mean(screens.std(axis=(1, 2))))
    measured_corr = measured_correlation(screens, pitch)
    rms_tolerance = 0.05 * max(glass.rms_rad, 1e-6)
    checks.append({"check": "glass_rms", "value": measured_rms, "target": glass.rms_rad,
                   "limit": rms_tolerance,
                   "passed": abs(measured_rms - glass.rms_rad) <= rms_tolerance})
    corr_tolerance = max(0.35 * glass.correlation_m, 1.5 * pitch)
    checks.append({"check": "glass_correlation", "value": measured_corr,
                   "target": glass.correlation_m, "limit": corr_tolerance,
                   "passed": abs(measured_corr - glass.correlation_m) <= corr_tolerance})
    info.update({"glass_phase_rms_rad": measured_rms,
                 "glass_correlation_um": measured_corr * 1e6,
                 "glass_correlation_px": measured_corr / pitch,
                 "glass_phase_range": [float(screens.min()), float(screens.max())]})

    # (2) 两种标签路径在同一张含玻璃图样上必须一致（切换点附近）
    for name in ("circle", "grating"):
        z = geometry(shapes[name], config.far_threshold, config)["z_m"]
        phase = phase_screen(rng, n, pitch, glass.correlation_m, glass.rms_rad)
        field = field_after_glass(shapes[name], phase)
        far = postprocess(raw_intensity(field, z, config, "fraunhofer_fft"), config.log_alpha)
        asm = postprocess(raw_intensity(field, z, config, "scaled_angular_spectrum"),
                          config.log_alpha)
        value = image_metrics(far, asm)["mae"]
        checks.append({"check": "glass_far_asm_agreement", "shape": name,
                       "nf": config.far_threshold, "value": value, "limit": 5e-3,
                       "passed": value < 5e-3})

    # (3) 零相位屏时含玻璃真值必须逐位等于无玻璃图样（同一条代码路径）
    z = geometry(shapes["circle"], 1.0, config)["z_m"]
    zero = np.zeros((n, n), np.float32)
    clean = postprocess(np.abs(scaled_field(shapes["circle"], z, config))**2, config.log_alpha)
    glassy = postprocess(raw_intensity(field_after_glass(shapes["circle"], zero), z, config,
                                       "scaled_angular_spectrum"), config.log_alpha)
    value = float(np.max(np.abs(clean - glassy)))
    checks.append({"check": "zero_glass_identity", "value": value, "limit": 1e-6,
                   "passed": value < 1e-6})

    # (4) 毛玻璃到底改了多少 + 散斑能否被探测网格分辨（信息性，不作判定）
    rows = []
    for name, aperture in shapes.items():
        geo = geometry(aperture, 1.0, config)
        z = geo["z_m"]
        phase = phase_screen(rng, n, pitch, glass.correlation_m, glass.rms_rad)
        clean = postprocess(np.abs(scaled_field(aperture, z, config))**2, config.log_alpha)
        glassy = postprocess(raw_intensity(field_after_glass(aperture, phase), z, config,
                                           "scaled_angular_spectrum"), config.log_alpha)
        metric = image_metrics(glassy, clean)
        mask_size = n * pitch
        rows.append({"shape": name, "D_mm": geo["D_m"] * 1e3, "z_m": z,
                     "detector_pitch_um": geo["detector_pitch_m"] * 1e6,
                     "halo_px": mask_size / glass.correlation_m,
                     "grain_px": mask_size / geo["D_m"],
                     "mae_clean_vs_glass": metric["mae"],
                     "corr_clean_vs_glass": metric["corr"]})
    info["glass_effect"] = rows
    info["median_grain_px"] = float(np.median([r["grain_px"] for r in rows]))
    info["median_halo_px"] = float(np.median([r["halo_px"] for r in rows]))
    info["median_residual_mae"] = float(np.median([r["mae_clean_vs_glass"] for r in rows]))
    if glass.rms_rad > 0:
        checks.append({"check": "glass_has_effect", "value": info["median_residual_mae"],
                       "limit": 1e-6,
                       "passed": info["median_residual_mae"] > 1e-6})

    _plot_glass_check(config, glass, shapes, rng, out / "glass_check.png")
    with (out / "glass_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"schema": 3, "config": config.to_dict(), "glass": glass.to_dict(),
            "nf_values": list(nf_values), "checks": checks, "info": info,
            "reference": "复振幅缩放角谱（含精确角谱色散修正），相位屏在物面",
            "passed": all(c["passed"] for c in checks),
            "note": "数值验收条件属于本次实验，不是普适物理边界。"}


def _plot_glass_check(config, glass, shapes, rng, path):
    names = list(shapes)[:4]
    fig, axes = plt.subplots(len(names), 5, figsize=(16, 3.1 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        aperture = shapes[name]
        geo = geometry(aperture, 1.0, config)
        z = geo["z_m"]
        phase = phase_screen(rng, config.grid, config.input_pitch_m,
                             glass.correlation_m, glass.rms_rad)
        clean = postprocess(np.abs(scaled_field(aperture, z, config))**2, config.log_alpha)
        glassy = postprocess(raw_intensity(field_after_glass(aperture, phase), z, config,
                                           "scaled_angular_spectrum"), config.log_alpha)
        residual = glassy - clean
        limit = float(np.abs(residual).max()) + 1e-12
        mae = image_metrics(glassy, clean)["mae"]
        panels = [(aperture, "gray", 0, 1, name),
                  (phase, "twilight", float(phase.min()), float(phase.max()),
                   "glass phase (rad)"),
                  (clean, "inferno", 0, 1, "clean (no glass)"),
                  (glassy, "inferno", 0, 1, "GT with glass (NF=1)"),
                  (residual, "RdBu_r", -limit, limit, f"ΔI = GT − clean\nMAE={mae:.4f}")]
        for col, (value, cmap, low, high, title) in enumerate(panels):
            image = axes[row, col].imshow(value, cmap=cmap, vmin=low, vmax=high,
                                          interpolation="nearest")
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")
            if col in (2, 3, 4):
                fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.03)
    fig.suptitle(f"glass check: correlation={glass.correlation_m*1e6:.0f} um, "
                 f"rms={glass.rms_rad:g} rad, "
                 f"{'shared' if glass.shared else 'per-sample'}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ============================================================
# 4. 生成数据
# ============================================================
def generate(config, glass, nf_values, samples, seed, out, save_clean=False):
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

    # 相位屏用独立 RNG：光阑序列与 v1 完全一致。
    glass_rng = np.random.RandomState(seed + GLASS_SEED_OFFSET)
    if glass.shared:
        shared = phase_screen(glass_rng, config.grid, config.input_pitch_m,
                              glass.correlation_m, glass.rms_rad)
        np.save(out / "glass_phase.npy", shared)
        phases = np.broadcast_to(shared, shape)
    else:
        phases = np.lib.format.open_memmap(out / "glass_phase.npy", mode="w+",
                                           dtype="float32", shape=shape)
        for i in range(samples):
            phases[i] = phase_screen(glass_rng, config.grid, config.input_pitch_m,
                                     glass.correlation_m, glass.rms_rad)
        phases.flush()
    np.savez(out / "splits.npz",
             **{k: np.asarray(v, dtype=np.int64) for k, v in splits.items()})

    groups = []
    for nf in nf_values:
        folder = out / group_name(nf)
        folder.mkdir()
        method = "fraunhofer_fft" if nf < config.far_threshold else "scaled_angular_spectrum"
        entry = {"name": group_name(nf), "nf": nf, "label_method": method, "folder": folder,
                 "patterns": np.lib.format.open_memmap(folder / "patterns.npy", mode="w+",
                                                       dtype="float32", shape=shape),
                 "clean": None}
        if save_clean:
            entry["clean"] = np.lib.format.open_memmap(folder / "patterns_clean.npy", mode="w+",
                                                       dtype="float32", shape=shape)
        groups.append(entry)

    # 外层按样本、内层按 NF：玻璃后的复振幅与 NF 无关，每个光阑只算一次。
    for i in range(samples):
        aperture = np.asarray(apertures[i], dtype=np.float64)
        field = field_after_glass(aperture, np.asarray(phases[i]))
        for entry in groups:
            z = geometry(aperture, entry["nf"], config)["z_m"]
            entry["patterns"][i] = postprocess(
                raw_intensity(field, z, config, entry["label_method"]), config.log_alpha)
            if entry["clean"] is not None:
                entry["clean"][i] = postprocess(
                    raw_intensity(aperture.astype(np.complex128), z, config,
                                  entry["label_method"]), config.log_alpha)
        if (i + 1) % max(1, samples // 10) == 0:
            print(f"  图样 {i + 1}/{samples}", flush=True)
    for entry in groups:
        entry["patterns"].flush()
        if entry["clean"] is not None:
            entry["clean"].flush()

    for entry in groups:
        _preview(config, apertures, phases, strategies, entry, glass)
        print(f"NF={entry['nf']:g} 完成：{entry['folder']}", flush=True)

    manifest = {
        "schema": 3, "status": "complete", "stage": "v2_glass",
        "config": config.to_dict(), "glass": glass.to_dict(),
        "glass_seed": seed + GLASS_SEED_OFFSET,
        "glass_note": ("相位屏是各向同性高斯随机场（rad），相关长度按自相关 1/e 半径定义；"
                       "逐样本随机时为 (samples, grid, grid)，--glass-shared 时是整个数据集"
                       "共用一块 (grid, grid)。"),
        "samples": samples, "seed": seed,
        "groups": [{"name": e["name"], "nf": e["nf"], "label_method": e["label_method"],
                    "has_clean": e["clean"] is not None} for e in groups],
        "split_counts": {k: len(v) for k, v in splits.items()},
        "D_definition": "非零像素包围盒较长跨度（含孔间距）；z=D^2/(lambda*NF)",
        "detector_grid": "x_j=(j-N/2)*lambda*z/(N*input_pitch)，与 v1 相同",
        "apertures_identical_to_v1": "同一 seed/samples 下 apertures.npy 与 v1 逐位相同",
        "note": "GT 含毛玻璃（物面薄相位屏）；patterns_clean.npy 是无玻璃理想图样（可选）。",
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def _preview(config, apertures, phases, strategies, entry, glass):
    """八类光阑各取一个：光阑 | 相位屏 | 无玻璃 | 含玻璃。"""
    fig, axes = plt.subplots(len(STRATEGIES), 4, figsize=(12, 2.4 * len(STRATEGIES)),
                             squeeze=False)
    for row, strategy in enumerate(STRATEGIES):
        i = strategies.index(strategy)
        aperture = np.asarray(apertures[i], dtype=np.float64)
        phase = np.asarray(phases[i])
        z = geometry(aperture, entry["nf"], config)["z_m"]
        clean = postprocess(raw_intensity(aperture.astype(np.complex128), z, config,
                                          entry["label_method"]), config.log_alpha)
        glassy = np.asarray(entry["patterns"][i])
        panels = [(aperture, "gray", 0, 1, strategy),
                  (phase, "twilight", float(phase.min()), float(phase.max()), "glass phase"),
                  (clean, "inferno", 0, 1, "clean"),
                  (glassy, "inferno", 0, 1, f"NF={entry['nf']:g} GT (with glass)")]
        for col, (value, cmap, low, high, title) in enumerate(panels):
            axes[row, col].imshow(value, cmap=cmap, vmin=low, vmax=high,
                                  interpolation="nearest")
            axes[row, col].set_title(title, fontsize=8)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(entry["folder"] / "preview.png", dpi=100)
    plt.close(fig)


# ============================================================
# 5. 入口
# ============================================================
def add_glass_args(parser):
    parser.add_argument("--glass-preset", choices=tuple(GLASS_PRESETS), default="moderate",
                        help="毛玻璃预设：weak / moderate / strong / groundglass")
    parser.add_argument("--glass-correlation-um", type=float, default=None,
                        help="相位相关长度（µm）；给了就覆盖预设（预设按掩模尺寸的比例给）")
    parser.add_argument("--glass-rms-rad", type=float, default=None,
                        help="相位起伏 rms（弧度）；给了就覆盖预设")
    parser.add_argument("--glass-shared", action="store_true",
                        help="整个数据集共用一块毛玻璃（同一块实体玻璃），默认逐样本随机")


def glass_from_args(args, config):
    preset = GLASS_PRESETS[args.glass_preset]
    mask_size = config.grid * config.input_pitch_m
    correlation = (args.glass_correlation_um * 1e-6 if args.glass_correlation_um is not None
                   else preset["correlation_ratio"] * mask_size)
    rms = args.glass_rms_rad if args.glass_rms_rad is not None else preset["rms_rad"]
    return GlassConfig(correlation_m=correlation, rms_rad=rms, shared=args.glass_shared)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_physics_args(parser)
    add_glass_args(parser)
    parser.add_argument("--samples", type=int, default=1000, help="原始光阑数，七组共用")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/v2")
    parser.add_argument("--save-clean", action="store_true",
                        help="额外保存无玻璃的理想图样 patterns_clean.npy（体积翻倍）")
    parser.add_argument("--skip-validation", action="store_true",
                        help="跳过生成前的数值验收（不建议）")
    args = parser.parse_args()
    if args.samples < 24:
        parser.error("--samples 至少为 24（每类 3 个光阑）")
    config = config_from_args(args)
    nf_values = check_nf(args.nf)
    glass = glass_from_args(args, config)
    if glass.correlation_m < 2 * config.input_pitch_m:
        print(f"[警告] 相关长度 {glass.correlation_m*1e6:.1f} µm 小于 2 个像素，"
              f"相位屏在网格上接近白噪声", flush=True)
    out = new_output(args.out)

    if not args.skip_validation:
        print("传播与毛玻璃验证 ...", flush=True)
        report = validate(config, glass, nf_values, out)
        write_json(out / "validation.json", report)
        if not report["passed"]:
            failed = [c for c in report["checks"] if not c["passed"]]
            raise SystemExit(f"验证未通过：{failed}\n未写入训练数据")
        print(f"验证通过：{sum(c['passed'] for c in report['checks'])}/"
              f"{len(report['checks'])} 项", flush=True)
        info = report["info"]
        print(f"  相位屏：rms={info['glass_phase_rms_rad']:.3f} rad, "
              f"相关长度={info['glass_correlation_um']:.0f} µm "
              f"({info['glass_correlation_px']:.1f} px)", flush=True)
        print(f"  玻璃扰动：中位 MAE={info['median_residual_mae']:.4f}；"
              f"散斑 grain≈{info['median_grain_px']:.2f} px, "
              f"halo≈{info['median_halo_px']:.1f} px", flush=True)
        if info["median_grain_px"] < 1.5:
            print("  [提示] grain < 1.5 px：扰动细节接近探测网格的 Nyquist，"
                  "逐像素学习会很困难；可加大相关长度或改用 weak 预设", flush=True)
    else:
        write_json(out / "validation.json", {"skipped": True})

    manifest = generate(config, glass, nf_values, args.samples, args.seed, out,
                        save_clean=args.save_clean)
    print(f"完成：{out}")
    print(f"划分：{manifest['split_counts']}")
    print(f"毛玻璃：相关长度 {glass.correlation_m*1e6:.0f} µm, rms {glass.rms_rad:g} rad, "
          f"{'共用一块' if glass.shared else '逐样本随机'}")


if __name__ == "__main__":
    main()
