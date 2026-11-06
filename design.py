"""目标图样 -> 用训练好的 physics+残差 网络当正向，反推光阑（本仓库唯一的 design 脚本）。

正向不再是解析公式，而是你自己训练出来的 `physics` 模型（固定传播层 + 残差 U-Net）：

    光阑 A  →  网络(physics + 残差)  →  预测图样  →  与目标比较  →  反传更新 A

网络是在「这套装置 + 这块毛玻璃」的数据上训出来的，所以它天然把玻璃（以及别的
系统性偏差）算进了正向——不需要知道 φ，也不需要改任何网络代码。

但要注意：网络只是在**训练分布附近**可信，而设计优化会主动把光阑推出那个分布。
所以必须用 `--check-glass` 把设计好的光阑放回真实模型（玻璃 + 缩放角谱）复算一遍。
不给 `--check-glass` 时，脚本会明确警告：屏幕上那个 corr 只是网络自己的说法。

用法：
    # 先 train.py 训练，得到 runs/.../nf_*/physics/best_model.pt
    python design.py --checkpoint runs/v1/nf_1/physics/best_model.pt --char Z \
        --out design_runs/net_Z

    # 有毛玻璃数据时，把真实相位屏当复核模型
    python design.py --checkpoint runs/v2_weak1/nf_1/physics/best_model.pt --char Z \
        --check-glass data/v2_weak1/glass_phase.npy --out design_runs/net_Z

    # 自己的目标图
    python design.py --checkpoint ... --target inputs/star.png --out design_runs/net_star

输出：target.npy / aperture.npy / aperture.png / pattern_pred.npy / pattern_fit.png /
      pattern_roi.png / target_roi.png / history.npy / comparison.png / result.json
      （给了 --check-glass 时还有 pattern_check_intensity.npy / pattern_check_roi.png）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from generate import PhysicsConfig
from generate_v2 import field_after_glass, raw_intensity
from train import ForwardModel

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

DISPLAY_ALPHA = 100.0


# ============================================================================
# 1. 目标图样与场景（与 v1 的 design 完全一致的约定）
# ============================================================================
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def crop_content(arr: np.ndarray, thresh: float = 0.05) -> np.ndarray:
    """裁掉四周纯黑边，只留图案本身；图案贴边时原样返回。"""
    rows = np.nonzero(arr.max(axis=1) > thresh)[0]
    cols = np.nonzero(arr.max(axis=0) > thresh)[0]
    if not len(rows) or not len(cols):
        return arr
    return arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def load_image(path: Path, invert: bool, auto_invert: bool = True,
               threshold: float | None = None) -> np.ndarray:
    """读 PNG/JPG/BMP/WEBP/TIFF/NPY，返回 [0,1] 灰度 float32（黑底亮图）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到目标图样：{path}")
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        img = Image.open(path)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            background = Image.new("RGBA", img.size, (0, 0, 0, 255))
            img = Image.alpha_composite(background, img).convert("L")
        else:
            img = img.convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.clip(arr, 0, 1)
    if invert:
        arr = 1.0 - arr
    elif auto_invert and float(np.median(arr)) > 0.5:
        arr = 1.0 - arr
        print("  [auto] 这张图看起来是白底黑图，已自动反色（--no-auto-invert 可关闭）")
    if arr.max() <= 1e-6:
        raise ValueError(f"目标图样全黑（没有任何亮像素）：{path}")
    arr = arr / arr.max()
    if threshold is not None:
        if not 0 < threshold < 1:
            raise ValueError("--threshold 必须在 (0, 1) 之间")
        arr = (arr > threshold).astype(np.float32)
        if arr.max() <= 0:
            raise ValueError(f"--threshold={threshold} 把整张图都切没了，换个更小的阈值")
    return crop_content(arr)


def fit_square(arr: np.ndarray, size: int, pad: int = 6) -> np.ndarray:
    """等比缩放到 size×size 的方框里并居中，空白补黑（不拉伸变形）。"""
    arr = np.clip(arr, 0, 1).astype(np.float32)
    h, w = arr.shape
    inner = max(1, size - 2 * pad)
    scale = inner / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = Image.fromarray((arr * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    out = np.zeros((size, size), np.float32)
    r0, c0 = (size - nh) // 2, (size - nw) // 2
    out[r0:r0 + nh, c0:c0 + nw] = np.asarray(resized, dtype=np.float32) / 255.0
    return out


def glyph_image(char: str, size: int, font_path: str) -> np.ndarray:
    """把字符渲染成 size×size 的白字黑底图，保持长宽比并居中。"""
    if len(char) != 1:
        raise ValueError("--char 只接受单个字符，多字符请用 --target 传图片")
    font = ImageFont.truetype(font_path, int(size * 1.6))
    canvas = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(canvas).text((size * 2, size * 2), char, fill=255, font=font, anchor="mm")
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    rows = np.nonzero(arr.max(axis=1) > 0.15)[0]
    cols = np.nonzero(arr.max(axis=0) > 0.15)[0]
    if not len(rows) or not len(cols):
        raise ValueError(f"字体渲染不出字符 {char!r}")
    crop = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    box = int(size * 0.86)
    h = box
    w = max(1, int(round(crop.shape[1] * h / crop.shape[0])))
    if w > box:
        w = box
        h = max(1, int(round(crop.shape[0] * w / crop.shape[1])))
    resized = Image.fromarray((crop * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)
    out = np.zeros((size, size), np.float32)
    r0, c0 = (size - h) // 2, (size - w) // 2
    out[r0:r0 + h, c0:c0 + w] = np.asarray(resized, dtype=np.float32) / 255.0
    return out


@dataclass
class Scene:
    grid: int
    roi: tuple[int, int, int, int]          # r0, r1, c0, c1
    target: np.ndarray                       # (B,B) ROI 内的目标（含留白）
    glyph: np.ndarray                        # (S,S) 目标图样本身
    canvas: np.ndarray                       # (N,N) 目标在整幅画布上的位置
    mask: np.ndarray                         # (N,N) ROI=1
    modulation: str
    source: str


def build_scene(args) -> Scene:
    grid, size, margin = args.grid, args.target_size, args.margin
    box = size + 2 * margin
    if box > grid // 2:
        raise ValueError("target_size + 2*margin 不能超过 grid/2（目标必须能装进一个象限）")
    if args.target is not None:
        img = fit_square(load_image(Path(args.target), args.invert, args.auto_invert,
                                    args.threshold), size)
        source = str(args.target)
    else:
        img = glyph_image(args.char, size, args.font)
        source = f"char:{args.char}"
    if args.modulation == "phase" or args.roi_mode == "center":
        r0 = (grid - box) // 2
        c0 = (grid - box) // 2
    else:
        r0 = grid // 2 - box          # 右上象限
        c0 = grid // 2
    roi = (r0, r0 + box, c0, c0 + box)
    canvas = np.zeros((grid, grid), np.float32)
    canvas[r0 + margin:r0 + margin + size, c0 + margin:c0 + margin + size] = img
    roi_target = np.zeros((box, box), np.float32)
    roi_target[margin:margin + size, margin:margin + size] = img
    mask = np.zeros((grid, grid), np.float32)
    mask[roi[0]:roi[1], roi[2]:roi[3]] = 1.0
    return Scene(grid, roi, roi_target, img, canvas, mask, args.modulation, source)


def forward(field: torch.Tensor) -> torch.Tensor:
    """纯 FFT —— 只用于初始化时挑载频方向，不参与最终设计。"""
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1)), norm="ortho"), dim=(-2, -1))


def aperture_from_theta(theta: torch.Tensor, modulation: str, hard: bool, ste: bool):
    if modulation == "phase":
        return torch.exp(1j * theta)
    soft = torch.sigmoid(theta)
    if modulation == "gray" or not hard:
        return soft
    binary = (soft > 0.5).to(soft.dtype)
    if ste:
        return binary + soft - soft.detach()
    return binary


def init_theta(scene: Scene, seed: int, device: torch.device) -> torch.Tensor:
    n = scene.grid
    gen = torch.Generator(device=device).manual_seed(seed)
    if scene.modulation == "phase":
        theta = torch.rand(n, n, generator=gen, device=device) * 2 * math.pi
        return theta.clone().requires_grad_(True)
    r0, r1, c0, c1 = scene.roi
    fy = (r0 + r1 - 1) / 2 - n / 2
    fx = (c0 + c1 - 1) / 2 - n / 2
    yy = torch.arange(n, device=device) - n // 2
    xx = torch.arange(n, device=device) - n // 2
    ramp = 2 * math.pi * (fx * xx[None, :] + fy * yy[:, None]) / n
    noise = torch.rand(n, n, generator=gen, device=device) * 2 * math.pi
    best, best_energy = None, -1.0
    for sign in (1.0, -1.0):
        candidate = 0.5 + 0.5 * torch.cos(sign * ramp + noise)
        energy = float((forward(candidate).abs() ** 2
                        * torch.as_tensor(scene.mask, device=device)).sum())
        if energy > best_energy:
            best, best_energy = candidate, energy
    return torch.logit(best.clamp(1e-3, 1 - 1e-3)).clone().requires_grad_(True)


@dataclass
class DesignConfig:
    steps: int = 1500
    lr: float = 0.05
    hard_fraction: float = 0.55
    w_outside: float = 0.02
    w_binary: float = 0.002
    w_area: float = 2.0
    fill_target: float = 0.4
    theta_clip: float = 5.0


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def twin_origin(scene: Scene) -> tuple[int, int]:
    n = scene.grid
    r0, r1, c0, c1 = scene.roi
    return int(n - r1 + 1), int(n - c1 + 1)


def display(intensity: np.ndarray, peak: float) -> np.ndarray:
    value = np.log1p(DISPLAY_ALPHA * intensity / max(peak, 1e-12))
    return value / math.log1p(DISPLAY_ALPHA)


def save_aperture_png(path: Path, aperture: np.ndarray, modulation: str) -> None:
    if modulation == "phase":
        phase = np.angle(aperture)
        cmap = plt.get_cmap("twilight")
        rgba = cmap((phase + np.pi) / (2 * np.pi))
        Image.fromarray((rgba * 255).astype(np.uint8)).save(path)
    else:
        Image.fromarray((np.clip(aperture.real, 0, 1) * 255).astype(np.uint8)).save(path)


# ============================================================================
# 2. 训练好的模型当正向
# ============================================================================
def load_forward_model(checkpoint_path, device):
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "physics":
        raise SystemExit("这个 checkpoint 不是 physics 模型（纯 U-Net 没有物理层，不能当正向用）")
    config = PhysicsConfig(**checkpoint["physics_config"])
    model = ForwardModel(checkpoint["kind"], checkpoint["base_ch"], config.log_alpha,
                         propagator=checkpoint["propagator"], physics_config=config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"config": config, "nf": float(checkpoint["nf"]),
                   "propagator": checkpoint["propagator"],
                   "condition_mean": np.asarray(checkpoint["condition_mean"], np.float64),
                   "condition_std": np.asarray(checkpoint["condition_std"], np.float64),
                   "epoch": checkpoint.get("epoch")}


def condition_for(z_m, config, info, device):
    """与训练时完全一致的标准化：log(z), log(观察面像素间距)。"""
    pitch = config.wavelength_m * z_m / (config.grid * config.input_pitch_m)
    normalized = (np.log(np.array([z_m, pitch])) - info["condition_mean"]) / info["condition_std"]
    return (torch.tensor(normalized, dtype=torch.float32, device=device)[None],
            torch.tensor([z_m], dtype=torch.float64, device=device))


def predict(model, aperture, condition, z, alpha):
    """网络输出（log 压缩、峰值归一化）-> 反压缩成强度。"""
    pred = model(aperture[None, None], condition, z)[0][0, 0].clamp(0.0, 1.0)
    return (torch.expm1(pred * math.log1p(alpha)) / alpha).clamp_min(0.0)


# ============================================================================
# 3. 优化
# ============================================================================
def optimize(scene, config, seed, device, model, condition, z, alpha):
    r0, r1, c0, c1 = scene.roi
    target = torch.as_tensor(scene.target, device=device)
    mask = torch.as_tensor(scene.mask, device=device)
    roi_slice = (slice(r0, r1), slice(c0, c1))
    roi_mask = mask[roi_slice]
    target_norm = target.norm().clamp_min(1e-12)

    theta = init_theta(scene, seed, device)
    optimiser = torch.optim.Adam([theta], lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, config.steps,
                                                           eta_min=config.lr * 0.1)
    history = {"step": [], "data": [], "outside": [], "binary": []}
    started = time.time()

    for step in range(config.steps):
        frac = step / max(config.steps - 1, 1)
        hard = scene.modulation == "binary" and frac >= config.hard_fraction
        aperture = aperture_from_theta(theta, scene.modulation, hard, ste=hard)
        amplitude = predict(model, aperture, condition, z, alpha).sqrt()
        scale = (amplitude[roi_slice].norm() / target_norm).clamp_min(1e-12)
        normalized = amplitude / scale
        diff = normalized[roi_slice] - target
        data = (roi_mask * diff ** 2).sum() / roi_mask.sum()
        outside = ((1 - mask) * normalized ** 2).sum() / mask.sum()
        loss = data + config.w_outside * outside
        if scene.modulation != "phase":
            real = aperture.real if torch.is_complex(aperture) else aperture
            loss = loss + config.w_binary * (real * (1 - real)).mean()
            loss = loss + config.w_area * (real.mean() - config.fill_target) ** 2
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        scheduler.step()
        if scene.modulation != "phase":
            with torch.no_grad():
                theta.clamp_(-config.theta_clip, config.theta_clip)
        if step % 25 == 0 or step == config.steps - 1:
            history["step"].append(step)
            history["data"].append(float(data.detach()))
            history["outside"].append(float(outside.detach()))
            real = aperture.real if torch.is_complex(aperture) else aperture
            history["binary"].append(float((real * (1 - real)).mean().detach()))

    with torch.no_grad():
        aperture = aperture_from_theta(theta, scene.modulation,
                                       hard=scene.modulation == "binary", ste=False)
        intensity = predict(model, aperture, condition, z, alpha)
        amplitude = intensity.sqrt()
        scale = (amplitude[roi_slice].norm() / target_norm).clamp_min(1e-12)
        diff = amplitude[roi_slice] / scale - target
        score = float((roi_mask * diff ** 2).sum() / roi_mask.sum())
    return {"aperture": aperture.detach().cpu(), "intensity": intensity.detach().cpu(),
            "score": score, "history": history, "elapsed_s": time.time() - started}


# ============================================================================
# 4. 指标 / 真值复核 / 可视化
# ============================================================================
def roi_amplitude(scene, intensity):
    r0, r1, c0, c1 = scene.roi
    amplitude = np.sqrt(np.maximum(intensity[r0:r1, c0:c1], 0))
    return amplitude * (np.linalg.norm(scene.target)
                        / max(float(np.linalg.norm(amplitude)), 1e-12))


def metrics(scene, aperture, intensity):
    r0, r1, c0, c1 = scene.roi
    roi_intensity = intensity[r0:r1, c0:c1]
    fit = roi_amplitude(scene, intensity)
    degenerate = float(np.linalg.norm(np.sqrt(np.maximum(roi_intensity, 0)))) < 1e-9
    tr0, tc0 = twin_origin(scene)
    window = (slice(tr0, tr0 + (r1 - r0)), slice(tc0, tc0 + (c1 - c0)))
    twin = intensity[window][::-1, ::-1] / max(float(roi_intensity.max()), 1e-12)
    roi_scaled = roi_intensity / max(float(roi_intensity.max()), 1e-12)
    total = float(intensity.sum())
    is_phase = scene.modulation == "phase"
    return {"roi": [int(v) for v in scene.roi], "degenerate": bool(degenerate),
            "corr_amplitude": pearson(fit, scene.target),
            "mae_amplitude": float(np.abs(fit - scene.target).mean()),
            "corr_twin": pearson(twin, roi_scaled),
            "roi_energy_fraction": float(roi_intensity.sum() / total),
            "aperture_fill_fraction": (float((np.real(aperture) > 0.5).mean())
                                       if not is_phase else 1.0)}


def check_through_glass(scene, aperture, phase, config, z_m):
    """把光阑放回真实解析模型（玻璃 + 缩放角谱）复算；phase=0 就是无玻璃。"""
    phase = np.asarray(phase, np.float32)
    if phase.shape != (config.grid, config.grid):
        raise SystemExit(f"--check-glass 需要 ({config.grid},{config.grid}) 的相位屏，"
                         f"实际是 {phase.shape}")
    intensity = raw_intensity(field_after_glass(aperture, phase), z_m, config,
                              "scaled_angular_spectrum")
    r0, r1, c0, c1 = scene.roi
    fit = roi_amplitude(scene, intensity)
    result = {"corr_amplitude": pearson(fit, scene.target),
              "mae_amplitude": float(np.abs(fit - scene.target).mean()),
              "roi_energy_fraction": float(intensity[r0:r1, c0:c1].sum()
                                           / max(float(intensity.sum()), 1e-30))}
    return result, intensity


def render_net(path, scene, aperture, intensity, metrics_, history, tag, check=None):
    r0, r1, c0, c1 = scene.roi
    roi_peak = max(float(intensity[r0:r1, c0:c1].max()), 1e-12)
    is_phase = scene.modulation == "phase"
    fig = plt.figure(figsize=(17.4, 8.6))
    gs = fig.add_gridspec(2, 3, left=0.033, right=0.975, top=0.85, bottom=0.10,
                          wspace=0.17, hspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(scene.canvas, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0,
                           fill=False, ec="#38e0ff", lw=1.6, ls="--"))
    ax.set_title("① 目标图样（蓝框 = ROI）", fontsize=12)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(display(intensity[r0:r1, c0:c1], roi_peak), cmap="inferno", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_title(f"② 网络预测的图样（ROI）  corr={metrics_['corr_amplitude']:.3f}", fontsize=12)

    ax = fig.add_subplot(gs[0, 2])
    if is_phase:
        ax.imshow(np.angle(aperture), cmap="twilight", interpolation="nearest")
        ax.set_title("③ 反设计光阑（相位）", fontsize=12)
    else:
        ax.imshow(np.real(aperture), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("③ 反设计光阑（白 = 透光）", fontsize=12)

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(display(intensity, roi_peak), cmap="inferno", vmin=0, vmax=1,
              interpolation="nearest")
    ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0,
                           fill=False, ec="#38e0ff", lw=1.6))
    ax.set_title("④ 全视场（网络预测，log 显示）", fontsize=12)

    ax = fig.add_subplot(gs[1, 1])
    if check is None:
        error = np.abs(scene.target - roi_amplitude(scene, intensity))
        title = "⑤ |目标 − 网络预测|（振幅）"
    else:
        error = np.abs(scene.target - roi_amplitude(scene, check))
        title = f"⑤ |目标 − 真实模型|（振幅）  corr={metrics_['check_corr']:.3f}"
    image = ax.imshow(error, cmap="magma", interpolation="nearest",
                      vmin=0, vmax=max(0.05, float(error.max())))
    ax.set_title(title, fontsize=12)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)

    ax = fig.add_subplot(gs[1, 2])
    steps = np.asarray(history["step"])
    ax.semilogy(steps, history["data"], color="#d94801", lw=1.6, label="ROI 拟合误差")
    ax.semilogy(steps, history["outside"], color="#4c78a8", lw=1.2, ls="--", label="ROI 外能量")
    ax.set_xlabel("迭代步", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("⑥ 收敛曲线", fontsize=12)

    for axes in fig.axes:
        axes.set_xticks([])
        axes.set_yticks([])
    fig.text(0.033, 0.945, tag, fontsize=13, fontweight="bold", ha="left")
    line = (f"网络预测 corr = {metrics_['corr_amplitude']:.3f}    "
            f"MAE = {metrics_['mae_amplitude']:.3f}    "
            f"ROI 能量占比 = {metrics_['roi_energy_fraction']*100:.1f}%")
    if check is not None:
        line += (f"      |      真实模型 corr = {metrics_['check_corr']:.3f}    "
                 f"MAE = {metrics_['check_mae']:.3f}")
    fig.text(0.033, 0.915, line, fontsize=9.5, color="#444444", ha="left")
    fig.text(0.033, 0.030,
             "正向是训练出来的 physics+残差 网络；网络只在训练分布附近可信，"
             "设计会把光阑推出分布，所以务必用 --check-glass 复核。",
             fontsize=8.8, color="#666666", ha="left")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


# ============================================================================
# 5. 入口
# ============================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="目标图样 -> 用训练好的 physics+残差 网络当正向反推光阑",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="train.py 的 best_model.pt（physics）")
    parser.add_argument("--char", default="Z")
    parser.add_argument("--target", default=None)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--no-auto-invert", dest="auto_invert", action="store_false")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--font", default=r"C:\Windows\Fonts\arialbd.ttf")
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--margin", type=int, default=None)
    parser.add_argument("--modulation", choices=["binary", "gray", "phase"], default="binary")
    parser.add_argument("--roi-mode", choices=["auto", "center"], default="auto")
    parser.add_argument("--z", type=float, default=None,
                        help="传播距离 (m)；默认由 checkpoint 的 NF 和整块掩模尺寸算出")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--check-glass", default=None,
                        help="真实玻璃的 (grid,grid) 相位 npy：设计完用它复算；"
                             "给全 0 的相位屏就是“无玻璃”复核")
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"输出目录已存在：{out}（重跑请换名）")
    out.mkdir(parents=True)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, info = load_forward_model(args.checkpoint, device)
    config = info["config"]
    z_m = args.z or (config.grid * config.input_pitch_m) ** 2 / (
        config.wavelength_m * info["nf"])
    condition, z = condition_for(z_m, config, info, device)
    print(f"  模型：{args.checkpoint}（epoch {info['epoch']}, propagator={info['propagator']}）",
          flush=True)
    print(f"  几何：NF={info['nf']:g}, z={z_m:.4f} m, grid={config.grid}", flush=True)

    scene_args = argparse.Namespace(**vars(args))
    scene_args.grid = config.grid
    if scene_args.target_size is None:
        scene_args.target_size = max(8, int(round(config.grid * 0.375)))
    if scene_args.margin is None:
        scene_args.margin = max(2, config.grid // 32)
    scene = build_scene(scene_args)

    best, runs = None, []
    for i in range(args.restarts):
        result = optimize(scene, DesignConfig(steps=args.steps, lr=args.lr),
                          args.seed + i, device, model, condition, z, config.log_alpha)
        runs.append({"seed": args.seed + i, "score": result["score"],
                     "elapsed_s": result["elapsed_s"]})
        print(f"  restart {i} (seed={args.seed + i}): score={result['score']:.5f} "
              f"({result['elapsed_s']:.1f}s)", flush=True)
        if best is None or result["score"] < best["score"]:
            best = result

    aperture = best["aperture"].numpy()
    intensity = best["intensity"].numpy()
    np.maximum(intensity, 0, out=intensity)
    metrics_ = metrics(scene, aperture, intensity)
    r0, r1, c0, c1 = scene.roi

    check_metrics, check_intensity = None, None
    if args.check_glass:
        check_metrics, check_intensity = check_through_glass(
            scene, aperture, np.load(args.check_glass), config, z_m)
        metrics_["check_corr"] = check_metrics["corr_amplitude"]
        metrics_["check_mae"] = check_metrics["mae_amplitude"]

    np.save(out / "target.npy", scene.canvas)
    np.save(out / "aperture.npy", aperture)
    np.save(out / "pattern_pred.npy", intensity)
    save_aperture_png(out / "aperture.png", aperture, scene.modulation)
    roi_peak = float(intensity[r0:r1, c0:c1].max())
    Image.fromarray((display(intensity, roi_peak) * 255).astype(np.uint8)).save(
        out / "pattern_fit.png")
    Image.fromarray((display(intensity[r0:r1, c0:c1], roi_peak) * 255).astype(np.uint8)).save(
        out / "pattern_roi.png")
    Image.fromarray((scene.target * 255).astype(np.uint8)).save(out / "target_roi.png")
    if check_intensity is not None:
        np.save(out / "pattern_check_intensity.npy", check_intensity)
        peak = max(float(check_intensity[r0:r1, c0:c1].max()), 1e-12)
        Image.fromarray((display(check_intensity[r0:r1, c0:c1], peak) * 255)
                        .astype(np.uint8)).save(out / "pattern_check_roi.png")
    np.save(out / "history.npy", np.array([best["history"]["step"], best["history"]["data"],
                                           best["history"]["outside"]]))
    render_net(out / "comparison.png", scene, aperture, intensity, metrics_, best["history"],
               f"{scene.source} → {scene.modulation} 光阑（正向 = 训练好的 physics+残差 网络）",
               check_intensity)

    result = {"stage": "net_design", "checkpoint": str(args.checkpoint),
              "target": scene.source, "modulation": scene.modulation, "grid": scene.grid,
              "nf": info["nf"], "z_m": z_m,
              "metrics": {k: v for k, v in metrics_.items() if k not in ("roi",)},
              "check_glass": ({"path": args.check_glass,
                               "note": "用真实解析模型（玻璃 + 缩放角谱）复算同一块光阑；"
                                       "相位全 0 即无玻璃",
                               "metrics": check_metrics} if check_metrics else None),
              "design": {"steps": args.steps, "restarts": args.restarts, "seed": args.seed,
                         "best_score": best["score"], "runs": runs},
              "warnings": (["掩模退化成近似全黑/全空，本次结果不可用"]
                           if metrics_["degenerate"] else []),
              "note": ("正向是学出来的：网络对图样的预测误差会直接限制设计可信度；"
                       "务必用 --check-glass 在真实模型上复算。"),
              "command": "python design.py " + " ".join(sys.argv[1:])}
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                                allow_nan=False), encoding="utf-8")

    print(f"\n完成：{out}")
    print(f"  网络预测 corr = {metrics_['corr_amplitude']:.4f}   "
          f"MAE = {metrics_['mae_amplitude']:.4f}")
    if check_metrics is not None:
        print(f"  [check] 真实模型 corr = {check_metrics['corr_amplitude']:.4f}   "
              f"MAE = {check_metrics['mae_amplitude']:.4f}")
    else:
        print("  [警告] 没有 --check-glass：上面的 corr 只是网络自己的预测，不要当成真结果。")


if __name__ == "__main__":
    main()
