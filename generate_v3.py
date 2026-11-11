"""阶段三 v3 数据生成：光阑 + **彻底随机的二维相位矩阵** -> 衍射图样（GT）。

和 v2 的区别只有一处：v2 的相位屏是**有相关长度的高斯随机场**（模拟一块有颗粒度的
毛玻璃），v3 直接把它换成一个**逐像素独立同分布的随机相位矩阵**——相关长度 = 0，
散射角覆盖整个视场，是"最彻底"的扩散体。

    v2: φ = 高斯随机场(ℓ ≈ 300 µm)        → 散射光晕 ~8 px
    v3: φ[i,j] ~ U(-π, π]  i.i.d.          → 散射光晕 = 整个视场（等于 grid px）

真值定义、传播器、后处理、数据集格式全部沿用 v2：

    I(x) = | P_z[ e^{iφ} · A ](x) |²      → 峰值归一化 + log1p(100·I/peak)

- `P_z` 复用 `generate_v2.scaled_field`（缩放角谱的复数版本），与 v2 逐行一致；
- `train.py` / `evaluate.py` / `design.py` **一行都不用改**：输出目录与 `generate.py` /
  `generate_v2.py` 完全同构，只是 `glass_phase.npy` 换成了 `phase_matrix.npy`
  （`design.py --check-glass` 传这个路径即可）。

物理上要知道的一点：相关长度为 0 时"记忆效应"完全消失，远场是散斑，
对铺满视场的光阑来说图样几乎不携带光阑形状的信息。所以这个数据集适合当**压力测试**
（"在这种最彻底的扩散体下，物理层 + 残差还能做到什么程度"），
或者用来模拟一块真正的强随机相位片。

用法：

    # 一块随机相位矩阵，全数据集共用（主用例）
    python generate_v3.py --samples 1000 --glass-shared --out data/v3

    # 逐样本独立随机
    python generate_v3.py --samples 1000 --out data/v3_random

    # 只要一组 NF，顺带混入设计产物分布
    python generate_v3.py --samples 600 --nf 1 --phase-shared --hologram-fraction 0.2 --out data/v3_nf1

    # 换相位分布：高斯 / 二值(0,π) / 限幅均匀
    python generate_v3.py --samples 1000 --phase-shared --phase-dist gaussian --phase-rms-rad 1.5 --out data/v3_g
    python generate_v3.py --samples 1000 --phase-shared --phase-dist binary   --phase-pi-prob 0.5 --out data/v3_b

输出（目录必须不存在；manifest 最后写，status=complete 才能训练）：

    apertures.npy           (N, grid, grid) float32
    phase_matrix.npy        (N, grid, grid) 弧度；--phase-shared 时是 (grid, grid)
    splits.npz              train / val / test
    validation.json         随机相位矩阵与两种标签路径的数值验收
    phase_metrics.csv       每个标准形状的扰动强度
    phase_check.png         光阑 | 相位矩阵 | 无相位 | 含相位 | 残差
    manifest.json           config / phase / groups / split_counts
    nf_*/patterns.npy       该 NF 的含相位标签 GT
    nf_*/patterns_clean.npy 无相位理想图样（仅 --save-clean）
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

from generate import (STRATEGIES, PhysicsConfig, add_hologram_args, add_physics_args,
                      apply_holograms, check_nf, config_from_args, fraunhofer_raw,
                      geometry, group_name, hologram_from_args, image_metrics,
                      make_aperture, new_output, postprocess, split_ids,
                      standard_apertures, write_json)
# 传播器与 v2 完全共用（复振幅缩放角谱）
from generate_v2 import field_after_glass as field_after_phase   # A·e^{iφ}，与 v2 同一实现
from generate_v2 import raw_intensity, scaled_field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PHASE_SEED_OFFSET = 3000017      # 与光阑 / 毛玻璃的 RNG 分开，互不干扰
PHASE_FILE = "phase_matrix.npy"


# ============================================================
# 1. 彻底随机的二维相位矩阵
# ============================================================
@dataclass(frozen=True)
class PhaseConfig:
    """逐像素 i.i.d. 的相位矩阵；相关长度就是 1 个像素（没有相关长度）。"""

    dist: str = "uniform"        # uniform | uniform_range | gaussian | binary
    range_deg: float = 180.0     # uniform_range: φ ~ U(-range/2, +range/2)
    rms_rad: float = 2.0         # gaussian: φ ~ N(0, rms²)
    pi_prob: float = 0.5         # binary: P(φ=π)，否则 0
    shared: bool = False

    def __post_init__(self):
        if self.dist not in ("uniform", "uniform_range", "gaussian", "binary"):
            raise ValueError("phase.dist 只能是 uniform / uniform_range / gaussian / binary")
        if not 0 < self.range_deg <= 360:
            raise ValueError("--phase-range-deg 必须在 (0, 360] 内")
        if not np.isfinite(self.rms_rad) or self.rms_rad <= 0:
            raise ValueError("--phase-rms-rad 必须为正")
        if not 0 <= self.pi_prob <= 1:
            raise ValueError("--phase-pi-prob 必须在 [0,1] 内")

    def expected_rms(self):
        """该分布的理论标准差，用于验收。"""
        if self.dist == "uniform":
            return float(np.pi / np.sqrt(3))                     # U(-π,π]
        if self.dist == "uniform_range":
            half = np.deg2rad(self.range_deg) / 2
            return float(half / np.sqrt(3))                      # U(-a,a]
        if self.dist == "gaussian":
            return float(self.rms_rad)
        return float(np.pi * np.sqrt(self.pi_prob * (1 - self.pi_prob)))   # 0/π

    def to_dict(self):
        return asdict(self)


def phase_matrix(rng, grid, phase):
    """生成 grid×grid 的随机相位矩阵（弧度）；每个像素独立。

    默认 `uniform`：φ ~ U(-π, π]，e^{iφ} 在单位圆上均匀分布 —— 散射最彻底，
    远场退化成散斑。`gaussian` / `uniform_range` / `binary` 是强度可调的变体。
    """
    if phase.dist == "uniform":
        values = rng.uniform(-np.pi, np.pi, (grid, grid))
    elif phase.dist == "uniform_range":
        half = np.deg2rad(phase.range_deg) / 2
        values = rng.uniform(-half, half, (grid, grid))
    elif phase.dist == "gaussian":
        values = rng.normal(0.0, phase.rms_rad, (grid, grid))
    else:  # binary
        values = np.where(rng.rand(grid, grid) < phase.pi_prob, np.pi, 0.0)
    return values.astype(np.float32)


def lag1_correlation(phases):
    """检查"逐像素独立"：e^{iφ} 的相邻像素相关系数应接近 0。"""
    values = np.atleast_3d(np.asarray(phases, dtype=np.float64))
    corr = []
    for p in values:
        field = np.exp(1j * p)
        a = field[:, :-1].ravel()
        b = field[:, 1:].ravel()
        denominator = np.sqrt(np.abs(a).dot(np.abs(a)) * np.abs(b).dot(np.abs(b)))
        corr.append(abs(float(np.real(np.vdot(a, b))) / max(denominator, 1e-30)))
    return float(np.mean(corr))


def measured_stats(phases):
    values = np.atleast_2d(np.asarray(phases, dtype=np.float64))
    return {"rms_rad": float(values.std()),
            "mean_rad": float(values.mean()),
            "min_rad": float(values.min()),
            "max_rad": float(values.max())}


# ============================================================
# 2. 生成前的数值验收
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


def validate(config, phase, nf_values, out):
    """写训练数组之前：相位矩阵统计、逐像素独立性、两种标签路径、恒等性。"""
    rng = np.random.RandomState(20261106)
    checks, info = [], {}
    shapes = _shapes(config)
    n, pitch = config.grid, config.input_pitch_m

    # (1) 相位分布统计是否与请求一致（rms 取理论值）
    screens = np.stack([phase_matrix(rng, n, phase) for _ in range(8)])
    stats = measured_stats(screens.reshape(-1, n, n))
    expected = phase.expected_rms()
    tolerance = 0.05 * expected if expected > 0 else 1e-9
    checks.append({"check": "phase_rms", "value": stats["rms_rad"], "target": expected,
                   "limit": tolerance,
                   "passed": abs(stats["rms_rad"] - expected) <= tolerance})
    info.update({"phase_rms_rad": stats["rms_rad"], "phase_expected_rms_rad": expected,
                 "phase_mean_rad": stats["mean_rad"],
                 "phase_range_rad": [stats["min_rad"], stats["max_rad"]],
                 "phase_dist": phase.dist})

    # (2) 逐像素独立（e^{iφ} 的一阶相邻相关 ≈ 0）
    iid = lag1_correlation(screens)
    checks.append({"check": "phase_iid", "value": iid, "limit": 0.02, "passed": iid < 0.02})
    info["phase_lag1_correlation"] = iid

    # (3) 相位是幺正的：含相位 / 不含相位的远场总能量必须相同（Parseval）
    aperture = shapes["circle"].astype(np.float64)
    z = geometry(aperture, max(min(nf_values), config.far_threshold), config)["z_m"]
    reference = float(fraunhofer_raw(aperture, z, config).sum())
    modulated = float(fraunhofer_raw(field_after_phase(aperture, screens[0]), z, config).sum())
    value = abs(modulated - reference) / max(reference, 1e-30)
    checks.append({"check": "phase_unitary_energy", "value": value, "limit": 1e-9,
                   "passed": value < 1e-9})

    # (4) 传播器自身的数值收敛（补零加倍）。这里**不**判定"FFT vs ASM 在切换点一致"：
    #     随机相位把能量铺满整个频带，两种近似在切换点本来就会分开（4 阶相位可达几十 rad），
    #     而每个 NF 组内部只用一种方法，是自洽的；两者的差别只记录下来供参考。
    info["far_asm_mae_at_threshold"] = {}
    for name in ("circle", "grating"):
        z = geometry(shapes[name], config.far_threshold, config)["z_m"]
        screen = phase_matrix(rng, n, phase)
        field = field_after_phase(shapes[name], screen)
        far = postprocess(raw_intensity(field, z, config, "fraunhofer_fft"), config.log_alpha)
        asm = postprocess(raw_intensity(field, z, config, "scaled_angular_spectrum"),
                          config.log_alpha)
        info["far_asm_mae_at_threshold"][name] = image_metrics(far, asm)["mae"]

    for name in ("circle", "grating"):
        aperture = shapes[name]
        z = geometry(aperture, 1.0, config)["z_m"]
        field = field_after_phase(aperture, phase_matrix(rng, n, phase))
        base = postprocess(np.abs(scaled_field(field, z, config,
                                               pad_factor=config.pad_factor))**2,
                           config.log_alpha)
        refined = postprocess(np.abs(scaled_field(field, z, config,
                                                  pad_factor=config.pad_factor * 2))**2,
                              config.log_alpha)
        value = image_metrics(refined, base)["mae"]
        checks.append({"check": "phase_padding_convergence", "shape": name,
                       "value": value, "limit": 1e-2, "passed": value < 1e-2})

    # (5) 相位全 0 时含相位真值必须逐位等于无相位图样
    z = geometry(shapes["circle"], 1.0, config)["z_m"]
    zero = np.zeros((n, n), np.float32)
    clean = postprocess(np.abs(scaled_field(shapes["circle"], z, config))**2, config.log_alpha)
    same = postprocess(raw_intensity(field_after_phase(shapes["circle"], zero), z, config,
                                     "scaled_angular_spectrum"), config.log_alpha)
    value = float(np.max(np.abs(clean - same)))
    checks.append({"check": "zero_phase_identity", "value": value, "limit": 1e-6,
                   "passed": value < 1e-6})

    # (6) 随机相位到底改了多少 + 散斑尺度（信息性，同时防止退化）
    rows = []
    for name, aperture in shapes.items():
        geo = geometry(aperture, 1.0, config)
        z = geo["z_m"]
        screen = phase_matrix(rng, n, phase)
        clean = postprocess(np.abs(scaled_field(aperture, z, config))**2, config.log_alpha)
        scrambled = postprocess(raw_intensity(field_after_phase(aperture, screen), z, config,
                                              "scaled_angular_spectrum"), config.log_alpha)
        metric = image_metrics(scrambled, clean)
        mask_size = n * pitch
        rows.append({"shape": name, "D_mm": geo["D_m"] * 1e3, "z_m": z,
                     "detector_pitch_um": geo["detector_pitch_m"] * 1e6,
                     # 相关长度 = 1 像素 -> 散射光晕铺满整个视场
                     "halo_px": float(n),
                     "grain_px": mask_size / geo["D_m"],
                     "mae_clean_vs_phase": metric["mae"],
                     "corr_clean_vs_phase": metric["corr"]})
    info["phase_effect"] = rows
    info["median_grain_px"] = float(np.median([r["grain_px"] for r in rows]))
    info["median_residual_mae"] = float(np.median([r["mae_clean_vs_phase"] for r in rows]))
    # binary 且 π_prob∈{0,1} 时 e^{iφ} 是常数，图样不变 —— 那时不做"必须有扰动"的判定
    inert = phase.dist == "binary" and (phase.pi_prob == 0 or phase.pi_prob == 1)
    if not inert:
        checks.append({"check": "phase_has_effect", "value": info["median_residual_mae"],
                       "limit": 1e-6,
                       "passed": info["median_residual_mae"] > 1e-6})

    _plot_phase_check(config, phase, shapes, rng, out / "phase_check.png")
    with (out / "phase_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"schema": 4, "config": config.to_dict(), "phase": phase.to_dict(),
            "nf_values": list(nf_values), "checks": checks, "info": info,
            "reference": "复振幅缩放角谱（含精确角谱色散修正），相位矩阵在物面",
            "passed": all(c["passed"] for c in checks),
            "note": "数值验收条件属于本次实验，不是普适物理边界。"}


def _plot_phase_check(config, phase, shapes, rng, path):
    names = list(shapes)[:4]
    fig, axes = plt.subplots(len(names), 5, figsize=(16, 3.1 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        aperture = shapes[name]
        geo = geometry(aperture, 1.0, config)
        z = geo["z_m"]
        screen = phase_matrix(rng, config.grid, phase)
        clean = postprocess(np.abs(scaled_field(aperture, z, config))**2, config.log_alpha)
        scrambled = postprocess(raw_intensity(field_after_phase(aperture, screen), z, config,
                                              "scaled_angular_spectrum"), config.log_alpha)
        residual = scrambled - clean
        limit = float(np.abs(residual).max()) + 1e-12
        mae = image_metrics(scrambled, clean)["mae"]
        panels = [(aperture, "gray", 0, 1, name),
                  (screen, "twilight", float(screen.min()), float(screen.max()),
                   "random phase (rad)"),
                  (clean, "inferno", 0, 1, "clean (no phase)"),
                  (scrambled, "inferno", 0, 1, "GT with random phase (NF=1)"),
                  (residual, "RdBu_r", -limit, limit, f"ΔI = GT − clean\nMAE={mae:.4f}")]
        for col, (value, cmap, low, high, title) in enumerate(panels):
            image = axes[row, col].imshow(value, cmap=cmap, vmin=low, vmax=high,
                                          interpolation="nearest")
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")
            if col in (2, 3, 4):
                fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.03)
    fig.suptitle(f"random phase check: {phase.dist}, rms={phase.expected_rms():.2f} rad, "
                 f"correlation = 0 (i.i.d. per pixel), "
                 f"{'shared' if phase.shared else 'per-sample'}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ============================================================
# 3. 生成数据
# ============================================================
def generate(config, phase, nf_values, samples, seed, out, save_clean=False, hologram=None):
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

    # 混入「设计产物」分布（密集干涉条纹），独立 RNG，不影响光阑/相位矩阵序列
    hologram_ids = []
    if hologram:
        rng_h = np.random.RandomState(seed + hologram["seed_offset"])
        hologram_ids = apply_holograms(apertures, rng_h, config.grid, samples,
                                       hologram["fraction"], hologram["mode"], hologram["fill"])
        apertures.flush()
        print(f"  混入 {len(hologram_ids)} 个密集条纹掩模"
              f"（{len(hologram_ids) / samples * 100:.0f}%，mode={hologram['mode']}）", flush=True)

    # 相位矩阵用独立 RNG
    phase_rng = np.random.RandomState(seed + PHASE_SEED_OFFSET)
    if phase.shared:
        shared = phase_matrix(phase_rng, config.grid, phase)
        np.save(out / PHASE_FILE, shared)
        phases = np.broadcast_to(shared, shape)
    else:
        phases = np.lib.format.open_memmap(out / PHASE_FILE, mode="w+",
                                           dtype="float32", shape=shape)
        for i in range(samples):
            phases[i] = phase_matrix(phase_rng, config.grid, phase)
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

    # 外层按样本、内层按 NF：相位屏后的复振幅与 NF 无关，每个光阑只算一次
    for i in range(samples):
        aperture = np.asarray(apertures[i], dtype=np.float64)
        field = field_after_phase(aperture, np.asarray(phases[i]))
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
        _preview(config, apertures, phases, strategies, entry)
        print(f"NF={entry['nf']:g} 完成：{entry['folder']}", flush=True)

    manifest = {
        "schema": 4, "status": "complete", "stage": "v3_random_phase",
        "config": config.to_dict(), "phase": phase.to_dict(),
        "phase_seed": seed + PHASE_SEED_OFFSET,
        "phase_file": PHASE_FILE,
        "phase_note": ("逐像素独立同分布的随机相位矩阵（弧度），相关长度 = 1 个像素；"
                       "逐样本随机时为 (samples, grid, grid)，--phase-shared 时是 (grid, grid)。"
                       "设计脚本用 --check-glass 指向 phase_matrix.npy 即可。"),
        "samples": samples, "seed": seed,
        "hologram": ({"fraction": hologram["fraction"], "mode": hologram["mode"],
                      "fill": list(hologram["fill"]), "ids": hologram_ids}
                     if hologram else None),
        "hologram_note": ("这些编号的样本被替换成设计产物风格的密集条纹掩模。"
                          if hologram else "未混入密集条纹掩模。"),
        "groups": [{"name": e["name"], "nf": e["nf"], "label_method": e["label_method"],
                    "has_clean": e["clean"] is not None} for e in groups],
        "split_counts": {k: len(v) for k, v in splits.items()},
        "D_definition": "非零像素包围盒较长跨度（含孔间距）；z=D^2/(lambda*NF)",
        "detector_grid": "x_j=(j-N/2)*lambda*z/(N*input_pitch)，与 v1/v2 相同",
        "note": "GT 含彻底随机相位矩阵（物面）；patterns_clean.npy 是无相位理想图样（可选）。",
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def _preview(config, apertures, phases, strategies, entry):
    """八类光阑各取一个：光阑 | 相位矩阵 | 无相位 | 含相位。"""
    fig, axes = plt.subplots(len(STRATEGIES), 4, figsize=(12, 2.4 * len(STRATEGIES)),
                             squeeze=False)
    for row, strategy in enumerate(STRATEGIES):
        i = strategies.index(strategy)
        aperture = np.asarray(apertures[i], dtype=np.float64)
        screen = np.asarray(phases[i])
        z = geometry(aperture, entry["nf"], config)["z_m"]
        clean = postprocess(raw_intensity(aperture.astype(np.complex128), z, config,
                                          entry["label_method"]), config.log_alpha)
        scrambled = np.asarray(entry["patterns"][i])
        panels = [(aperture, "gray", 0, 1, strategy),
                  (screen, "twilight", float(screen.min()), float(screen.max()),
                   "random phase"),
                  (clean, "inferno", 0, 1, "clean"),
                  (scrambled, "inferno", 0, 1, f"NF={entry['nf']:g} GT (with phase)")]
        for col, (value, cmap, low, high, title) in enumerate(panels):
            axes[row, col].imshow(value, cmap=cmap, vmin=low, vmax=high,
                                  interpolation="nearest")
            axes[row, col].set_title(title, fontsize=8)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(entry["folder"] / "preview.png", dpi=100)
    plt.close(fig)


# ============================================================
# 4. 入口
# ============================================================
def add_phase_args(parser):
    parser.add_argument("--phase-dist", choices=("uniform", "uniform_range", "gaussian", "binary"),
                        default="uniform",
                        help="相位分布：uniform=U(-π,π]（最彻底）；uniform_range=U(-a,a)；"
                             "gaussian=N(0,rms²)；binary=0/π 两值")
    parser.add_argument("--phase-range-deg", type=float, default=180.0,
                        help="uniform_range 的半幅（度），φ ~ U(-range/2, +range/2)")
    parser.add_argument("--phase-rms-rad", type=float, default=2.0,
                        help="gaussian 的标准差（弧度）")
    parser.add_argument("--phase-pi-prob", type=float, default=0.5,
                        help="binary 分布里取 π 的概率（其余取 0）")
    parser.add_argument("--phase-shared", "--glass-shared", dest="phase_shared",
                        action="store_true",
                        help="整个数据集共用同一块随机相位矩阵（默认逐样本独立）")


def phase_from_args(args):
    return PhaseConfig(dist=args.phase_dist, range_deg=args.phase_range_deg,
                       rms_rad=args.phase_rms_rad, pi_prob=args.phase_pi_prob,
                       shared=args.phase_shared)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_physics_args(parser)
    add_phase_args(parser)
    add_hologram_args(parser)
    parser.add_argument("--samples", type=int, default=1000, help="原始光阑数，七组共用")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/v3")
    parser.add_argument("--save-clean", action="store_true",
                        help="额外保存无相位的理想图样 patterns_clean.npy（体积翻倍）")
    parser.add_argument("--skip-validation", action="store_true",
                        help="跳过生成前的数值验收（不建议）")
    args = parser.parse_args()
    if args.samples < 24:
        parser.error("--samples 至少为 24（每类 3 个光阑）")
    config = config_from_args(args)
    nf_values = check_nf(args.nf)
    phase = phase_from_args(args)
    out = new_output(args.out)

    if not args.skip_validation:
        print("随机相位矩阵与传播验证 ...", flush=True)
        report = validate(config, phase, nf_values, out)
        write_json(out / "validation.json", report)
        if not report["passed"]:
            failed = [c for c in report["checks"] if not c["passed"]]
            raise SystemExit(f"验证未通过：{failed}\n未写入训练数据")
        print(f"验证通过：{sum(c['passed'] for c in report['checks'])}/"
              f"{len(report['checks'])} 项", flush=True)
        info = report["info"]
        print(f"  相位矩阵：dist={info['phase_dist']}, rms={info['phase_rms_rad']:.3f} rad, "
              f"相邻像素相关={info['phase_lag1_correlation']:.4f}", flush=True)
        print(f"  扰动强度：中位 MAE={info['median_residual_mae']:.4f}；"
              f"散斑 grain≈{info['median_grain_px']:.2f} px，"
              f"散射光晕铺满整个视场（相关长度 = 1 像素）", flush=True)
        print("  [提示] 相关长度为 0 时远场是散斑、几乎不携带光阑形状信息；"
              "这组数据主要用于压力测试/强扩散体仿真", flush=True)
    else:
        write_json(out / "validation.json", {"skipped": True})

    manifest = generate(config, phase, nf_values, args.samples, args.seed, out,
                        save_clean=args.save_clean, hologram=hologram_from_args(args))
    print(f"完成：{out}")
    print(f"划分：{manifest['split_counts']}")
    print(f"随机相位：{phase.dist}, 理论 rms={phase.expected_rms():.3f} rad, "
          f"{'共用一块' if phase.shared else '逐样本独立'}")
    if manifest.get("hologram"):
        print(f"密集条纹掩模：{len(manifest['hologram']['ids'])} 个 "
              f"（mode={manifest['hologram']['mode']}）")


if __name__ == "__main__":
    main()
