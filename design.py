"""目标衍射图样 → 反设计光阑（计算全息 / 逆设计）。

正向问题是「光阑 → 衍射图样」（`generate.py`），单图逆向是「图样 → 光阑」
（`inverse.py`，要求恢复出**那个**真实光阑）。本脚本是第三种问题：

    给定一张想要的远场图样（例如字母 Z），反设计出一块光阑，
    使它的夫琅禾费衍射图样**就是**这张图。

光阑长什么样不做任何限制：解出来通常是一块干涉条纹状的二值掩模，
看起来像二维码 / 全息图，这是正常的——它只需要在远场重建目标。

两种调制方式（`--modulation`）：

- `binary`（默认）：二值振幅光阑，透过率 0/1，可打印/曝光加工。
  实振幅光阑的远场强度必然共轭对称 `I(k) = I(-k)`，所以不对称的 Z
  不可能单独出现在全视场正中央。做法是把目标放到**偏轴的一个象限**，
  优化时只在该 ROI 内比对；中心对称位置会自动出现一个旋转 180° 的
  孪生像（这是物理，不是误差）。只看 ROI（或挡掉其余区域）就得到单个 Z。
- `gray`：连续透过率，拟合上限，需要灰阶曝光或半色调。
- `phase`：纯相位光阑（`a = exp(iφ)`），复振幅不受共轭对称限制，
  Z 可以落在光轴中心、没有孪生像；适合 SLM / 刻蚀相位片。

损失（ROI 内，按 ROI 峰值归一化后的振幅）：

    L = Σ_ROI (|F|/peak − T)² / Σ_ROI 1
      + 0.02 · Σ_outside (|F|/peak)² / Σ_ROI 1     # 抑制 ROI 外杂散能量
      + 0.002 · mean(a(1−a))                        # 二值化（仅振幅模式）
      + 2 · (mean(a) − 0.4)²                        # 平均透过率，防止退化成全黑

二值模式先连续优化，后 45% 步数用 STE 直接优化 0/1 掩模；保存的图样
始终由**最终掩模**重新做一次 FFT 得到，不做任何“美化”。

用法：

    python design.py --char Z --out design_runs/letterZ
    python design.py --target my_pattern.png --out design_runs/custom
    python design.py --char Z --modulation phase --out design_runs/letterZ_phase
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

DISPLAY_ALPHA = 100.0  # 与 generate.postprocess 的 log1p(alpha*I/peak) 一致


# ----------------------------------------------------------------------------
# 1. 目标图样
# ----------------------------------------------------------------------------

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
    """读 PNG/JPG/BMP/WEBP/TIFF/NPY，返回 [0,1] 灰度 float32。

    - 彩色图转灰度；带 alpha 的 PNG 把透明部分当黑底合成；
    - 目标是「黑底亮图」，所以整幅中位亮度 > 0.5 视为白底黑图，自动反色
      （`--invert` 强制反色，`--no-auto-invert` 关闭自动判断）；
    - `--threshold` 把目标按亮度二值化成剪影（适合卡通/照片这类没有纯黑背景的图）。
    """
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
    grid = args.grid
    size = args.target_size
    margin = args.margin
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
        # 右上象限：ROI 中心相对画布中心偏移 (+g/4, -g/4)
        r0 = grid // 2 - box
        c0 = grid // 2
    roi = (r0, r0 + box, c0, c0 + box)

    canvas = np.zeros((grid, grid), np.float32)
    canvas[r0 + margin:r0 + margin + size, c0 + margin:c0 + margin + size] = img
    roi_target = np.zeros((box, box), np.float32)
    roi_target[margin:margin + size, margin:margin + size] = img
    mask = np.zeros((grid, grid), np.float32)
    mask[roi[0]:roi[1], roi[2]:roi[3]] = 1.0
    return Scene(grid, roi, roi_target, img, canvas, mask, args.modulation, source)


# ----------------------------------------------------------------------------
# 2. 正向算子与优化
# ----------------------------------------------------------------------------

def forward(field: torch.Tensor) -> torch.Tensor:
    """夫琅禾费衍射（unitariy FFT，只关心形状）。"""
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

    # 载频方向取能让 ROI 先有能量的那个符号
    best, best_e = None, -1.0
    for sign in (1.0, -1.0):
        a0 = 0.5 + 0.5 * torch.cos(sign * ramp + noise)
        energy = float((forward(a0).abs() ** 2 * torch.as_tensor(scene.mask, device=device)).sum())
        if energy > best_e:
            best, best_e = a0, energy
    theta = torch.logit(best.clamp(1e-3, 1 - 1e-3))
    return theta.clone().requires_grad_(True)


@dataclass
class DesignConfig:
    steps: int = 1500
    lr: float = 0.05
    hard_fraction: float = 0.55   # 之后切 STE 二值
    w_outside: float = 0.02
    w_binary: float = 0.002
    w_area: float = 2.0           # 平均透过率软约束，防止解退化成全黑/全开
    fill_target: float = 0.4
    theta_clip: float = 5.0       # 防止 sigmoid 饱和后梯度消失


def optimize(scene: Scene, config: DesignConfig, seed: int, device: torch.device):
    n = scene.grid
    r0, r1, c0, c1 = scene.roi
    target = torch.as_tensor(scene.target, device=device)
    mask = torch.as_tensor(scene.mask, device=device)
    roi_slice = (slice(r0, r1), slice(c0, c1))
    roi_mask = mask[roi_slice]
    target_norm = target.norm().clamp_min(1e-12)

    theta = init_theta(scene, seed, device)
    opt = torch.optim.Adam([theta], lr=config.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, config.steps, eta_min=config.lr * 0.1)
    history = {"step": [], "data": [], "outside": [], "binary": []}
    t0 = time.time()

    for step in range(config.steps):
        frac = step / max(config.steps - 1, 1)
        hard = scene.modulation == "binary" and frac >= config.hard_fraction
        aperture = aperture_from_theta(theta, scene.modulation, hard, ste=hard)

        amplitude = forward(aperture).abs()
        roi_amp = amplitude[roi_slice]
        # L2 归一化：让 ROI 内的场与目标同能量，损失只比较“形状”。
        # 用峰值归一化时，稀疏目标会让优化一路把场压暗（峰值跟着掉、归一化又放大外部
        # 能量），最后塌成全黑——线稿类目标必然踩到。
        scale = (roi_amp.norm() / target_norm).clamp_min(1e-12)
        norm = amplitude / scale
        diff = norm[roi_slice] - target
        data = (roi_mask * diff ** 2).sum() / roi_mask.sum()
        outside = ((1 - mask) * norm ** 2).sum() / mask.sum()
        loss = data + config.w_outside * outside
        if scene.modulation != "phase":
            real = aperture.real if torch.is_complex(aperture) else aperture
            loss = loss + config.w_binary * (real * (1 - real)).mean()
            # 数据项对整体透过率没有约束，用平均透过率把它钉在 fill_target 附近
            # （二值全息通常 ~50%）。
            loss = loss + config.w_area * (real.mean() - config.fill_target) ** 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if scene.modulation != "phase":
            with torch.no_grad():
                theta.clamp_(-config.theta_clip, config.theta_clip)

        if step % 25 == 0 or step == config.steps - 1:
            history["step"].append(step)
            history["data"].append(float(data.detach()))
            history["outside"].append(float(outside.detach()))
            history["binary"].append(float((aperture.real * (1 - aperture.real)).mean().detach()))

    elapsed = time.time() - t0
    with torch.no_grad():
        aperture = aperture_from_theta(theta, scene.modulation,
                                       hard=scene.modulation == "binary", ste=False)
        amplitude = forward(aperture).abs()
        scale = (amplitude[roi_slice].norm() / target_norm).clamp_min(1e-12)
        diff = amplitude[roi_slice] / scale - target
        score = float((roi_mask * diff ** 2).sum() / roi_mask.sum())
    return {
        "theta": theta.detach().cpu(),
        "aperture": aperture.detach().cpu(),
        "score": score,
        "history": history,
        "elapsed_s": elapsed,
    }


# ----------------------------------------------------------------------------
# 3. 指标
# ----------------------------------------------------------------------------

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def twin_origin(scene: Scene) -> tuple[int, int]:
    """ROI 对跖区域的左上角：像素 j 的镜像为 (N-j) mod N。

    N 为偶数时原点落在 (N/2, N/2) 像素上，镜像区与 ROI 同尺寸但相差
    180° 旋转，所以取回后还要 flip 一次。
    """
    n = scene.grid
    r0, r1, c0, c1 = scene.roi
    return int(n - r1 + 1), int(n - c1 + 1)


def evaluate(scene: Scene, aperture: np.ndarray, device: torch.device) -> dict:
    n = scene.grid
    r0, r1, c0, c1 = scene.roi
    if scene.modulation == "phase":
        field = torch.as_tensor(aperture, device=device).to(torch.complex64)
    else:
        field = torch.as_tensor(aperture, device=device).real
    with torch.no_grad():
        amplitude = forward(field).abs().cpu().numpy()
    intensity = amplitude ** 2

    roi_amp = amplitude[r0:r1, c0:c1]
    target = scene.target
    # 与优化一致：按 L2 能量把拟合场缩到目标的总能量再比较，
    # 这样 mae ≈ √score，且不会在拟合场接近 0 时被放大成假的高相关。
    roi_norm = float(np.linalg.norm(roi_amp))
    target_norm = float(np.linalg.norm(target))
    degenerate = roi_norm < 1e-9
    fit = np.zeros_like(roi_amp) if degenerate else roi_amp * (target_norm / roi_norm)

    # 对跖区域：实光阑必须与 ROI 逐点镜像 I(j) = I((N-j) mod N)
    tr0, tc0 = twin_origin(scene)
    twin = intensity[tr0:tr0 + (r1 - r0), tc0:tc0 + (c1 - c0)][::-1, ::-1] / max(
        intensity[r0:r1, c0:c1].max(), 1e-12)
    roi_int = intensity[r0:r1, c0:c1] / max(intensity[r0:r1, c0:c1].max(), 1e-12)

    total = float(intensity.sum())
    return {
        "roi": [int(v) for v in scene.roi],
        "roi_size": int(r1 - r0),
        "target_size": int(scene.glyph.shape[0]),
        "roi_box": int(scene.target.shape[0]),
        "peak_intensity": float(intensity[r0:r1, c0:c1].max()),
        "degenerate": bool(degenerate),
        "corr_amplitude": pearson(fit, target),
        "mae_amplitude": float(np.abs(fit - target).mean()),
        "corr_intensity": pearson(roi_int, target ** 2),
        "corr_twin": pearson(twin, roi_int) if scene.modulation != "phase" else None,
        "roi_energy_fraction": float(intensity[r0:r1, c0:c1].sum() / total),
        "dc_energy_fraction": float(intensity[n // 2 - 1:n // 2 + 1, n // 2 - 1:n // 2 + 1].sum() / total),
        "aperture_fill_fraction": float((aperture.real > 0.5).mean()) if scene.modulation != "phase" else 1.0,
        "aperture_binary": bool(scene.modulation == "binary"),
    }


# ----------------------------------------------------------------------------
# 4. 可视化
# ----------------------------------------------------------------------------

def display(intensity: np.ndarray, peak: float) -> np.ndarray:
    """与 data/v1 标签一致的 log1p 显示，按给定峰值归一化。"""
    value = np.log1p(DISPLAY_ALPHA * intensity / max(peak, 1e-12))
    return value / math.log1p(DISPLAY_ALPHA)


def save_aperture_png(path: Path, aperture: np.ndarray, modulation: str) -> None:
    if modulation == "phase":
        phase = np.angle(aperture)
        cmap = plt.get_cmap("twilight")
        rgba = cmap((phase + np.pi) / (2 * np.pi))
        Image.fromarray((rgba * 255).astype(np.uint8)).save(path)
    else:
        img = (np.clip(aperture.real, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(path)


def render(fig_path: Path, scene: Scene, aperture: np.ndarray, intensity: np.ndarray,
           metrics: dict, history: dict, tag: str) -> None:
    n = scene.grid
    r0, r1, c0, c1 = scene.roi
    roi_peak = max(float(intensity[r0:r1, c0:c1].max()), 1e-12)

    full = display(intensity, roi_peak)
    roi_fit = display(intensity[r0:r1, c0:c1], roi_peak)
    target = scene.canvas
    is_phase = scene.modulation == "phase"
    tr0, tc0 = twin_origin(scene)

    fig = plt.figure(figsize=(15.2, 9.8))
    gs = fig.add_gridspec(2, 3, left=0.035, right=0.975, top=0.865, bottom=0.085,
                          wspace=0.19, hspace=0.20)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(target, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax0.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0,
                            fill=False, ec="#38e0ff", lw=1.6, ls="--"))
    ax0.set_title("① 输入的目标衍射图样", fontsize=12, pad=8)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(roi_fit, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax1.set_title(f"② 拟合的衍射图样（ROI 裁剪）  corr = {metrics['corr_amplitude']:.3f}",
                  fontsize=12, pad=8)

    ax2 = fig.add_subplot(gs[0, 2])
    if is_phase:
        ax2.imshow(np.angle(aperture), cmap="twilight", interpolation="nearest")
        ax2.set_title("③ 反设计出的相位光阑 φ(x,y)", fontsize=12, pad=8)
    else:
        ax2.imshow(aperture.real, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax2.set_title("③ 反设计出的光阑（白 = 透光）", fontsize=12, pad=8)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(full, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ax3.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0,
                            fill=False, ec="#38e0ff", lw=1.6))
    if not is_phase:
        ax3.add_patch(Rectangle((tc0 - 0.5, tr0 - 0.5), c1 - c0, r1 - r0,
                                fill=False, ec="#7cff9b", lw=1.2, ls=":"))
        ax3.annotate("共轭孪生像", xy=(tc0 - 1, tr0 - 2), color="#0b3d16", fontsize=9,
                     ha="left", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.25", fc="#7cff9b", ec="none", alpha=0.85))
    ax3.annotate("目标", xy=(c0 - 1, r0 - 2), color="#4a2500", fontsize=9,
                 ha="left", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#38e0ff", ec="none", alpha=0.85))
    ax3.set_title("④ 全视场（log 显示，按 ROI 峰值归一化）", fontsize=12, pad=8)

    ax4 = fig.add_subplot(gs[1, 1])
    roi_amp_l2 = np.sqrt(intensity[r0:r1, c0:c1])
    roi_amp_l2 = roi_amp_l2 * (np.linalg.norm(scene.target) / max(np.linalg.norm(roi_amp_l2), 1e-12))
    error = np.abs(scene.target - roi_amp_l2)
    im = ax4.imshow(error, cmap="magma", vmin=0, vmax=max(0.05, float(error.max())),
                    interpolation="nearest")
    ax4.set_title("⑤ |目标 − 拟合| （振幅）", fontsize=12, pad=8)
    cb = plt.colorbar(im, ax=ax4, fraction=0.040, pad=0.02)
    cb.ax.tick_params(labelsize=8)

    ax5 = fig.add_subplot(gs[1, 2])
    steps = np.asarray(history["step"])
    ax5.semilogy(steps, history["data"], color="#d94801", lw=1.6, label="ROI 拟合误差")
    ax5.semilogy(steps, history["outside"], color="#4c78a8", lw=1.2, ls="--", label="ROI 外能量")
    ax5.set_xlabel("迭代步", fontsize=9)
    ax5.grid(alpha=0.25)
    ax5.legend(fontsize=8, frameon=False)
    ax5.set_title("⑥ 收敛曲线（对数）" + ("，0.55 处切二值 STE" if scene.modulation == "binary" else ""),
                  fontsize=11.5, pad=8)

    for ax in (ax0, ax1, ax2, ax3):
        ax.set_xticks([])
        ax.set_yticks([])

    note = ("实振幅光阑的远场必然满足 I(k) = I(−k)：ROI 里的目标会在对跖象限自动出现一个"
            "旋转 180° 的孪生像（这是物理，不是拟合误差）。",
            "只取 ROI，或在焦平面上用视场光阑挡掉其余区域，就得到单个 Z。") if not is_phase else (
        "相位光阑是复振幅，不受共轭对称限制：Z 直接落在光轴中心，没有孪生像。",
        "条纹尺度 ≈ 光阑像素尺度，实际制作需要 SLM 或亚微米刻蚀。")
    fig.text(0.035, 0.955, tag, fontsize=14, fontweight="bold", ha="left")
    fig.text(0.035, 0.922,
             f"振幅相关 corr = {metrics['corr_amplitude']:.3f}    "
             f"MAE = {metrics['mae_amplitude']:.3f}    "
             f"ROI 能量占比 = {metrics['roi_energy_fraction']*100:.1f}%    "
             + (f"孪生像与 ROI 的相关系数 = {metrics['corr_twin']:.4f}" if not is_phase else
                f"相位范围 = [−π, π]"),
             fontsize=9.5, color="#444444", ha="left")
    fig.text(0.035, 0.030, note[0], fontsize=8.8, color="#666666", ha="left")
    fig.text(0.035, 0.010, note[1], fontsize=8.8, color="#666666", ha="left")
    fig.savefig(fig_path, dpi=150, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 5. 入口
# ----------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="目标衍射图样 → 反设计光阑（夫琅禾费，二值振幅或纯相位）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--char", default="Z", help="要重建的字符（单字符）")
    parser.add_argument("--target", default=None,
                        help="目标图样 PNG/JPG/BMP/WEBP/TIFF/NPY；给了就忽略 --char")
    parser.add_argument("--invert", action="store_true", help="强制目标图反色（白底黑图时用）")
    parser.add_argument("--no-auto-invert", dest="auto_invert", action="store_false",
                        help="关闭白底自动反色（默认按整幅中位亮度自动判断）")
    parser.add_argument("--threshold", type=float, default=None,
                        help="把目标按亮度二值化成剪影，取值 0–1（卡通/照片这类没有纯黑背景的图很有用）")
    parser.add_argument("--font", default=r"C:\Windows\Fonts\arialbd.ttf", help="渲染 --char 的字体")
    parser.add_argument("--grid", type=int, default=256, help="光阑/图样网格 N")
    parser.add_argument("--target-size", type=int, default=96, help="目标图样在画布上的像素尺寸")
    parser.add_argument("--margin", type=int, default=8, help="ROI 内目标四周留白")
    parser.add_argument("--modulation", choices=["binary", "gray", "phase"], default="binary",
                        help="binary=二值振幅；gray=连续振幅；phase=纯相位")
    parser.add_argument("--roi-mode", choices=["auto", "center"], default="auto",
                        help="auto：振幅模式放右上象限、相位模式放中心")
    parser.add_argument("--steps", type=int, default=1500, help="每个 restart 的迭代步数")
    parser.add_argument("--restarts", type=int, default=2, help="随机初值 restart 次数")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam 学习率")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--pitch", type=float, default=10e-6, help="掩模像素间距 (m)，仅用于物理尺寸换算")
    parser.add_argument("--wavelength", type=float, default=532e-9, help="波长 (m)")
    parser.add_argument("--focal-length", type=float, default=0.5,
                        help="傅里叶透镜焦距 (m)，仅用于物理尺寸换算；0 表示不换算")
    parser.add_argument("--out", required=True, help="输出目录（必须不存在）")
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
    scene = build_scene(args)
    config = DesignConfig(steps=args.steps, lr=args.lr)

    best = None
    all_runs = []
    for i in range(args.restarts):
        result = optimize(scene, config, args.seed + i, device)
        all_runs.append({"seed": args.seed + i, "score": result["score"],
                         "elapsed_s": result["elapsed_s"]})
        print(f"  restart {i} (seed={args.seed + i}): score={result['score']:.5f} "
              f"({result['elapsed_s']:.1f}s)")
        if best is None or result["score"] < best["score"]:
            best = result

    aperture = best["aperture"].numpy()
    metrics = evaluate(scene, aperture, device)
    intensity = None
    field = (torch.as_tensor(aperture, device=device).to(torch.complex64)
             if scene.modulation == "phase" else torch.as_tensor(aperture, device=device).real)
    with torch.no_grad():
        intensity = forward(field).abs().cpu().numpy() ** 2

    # 物理尺寸换算
    geometry = {}
    if args.focal_length > 0:
        det_pitch = args.wavelength * args.focal_length / (scene.grid * args.pitch)
        fov = scene.grid * det_pitch
        fov_mm = fov * 1e3
        z_width = fov * scene.glyph.shape[0] / scene.grid
        geometry = {
            "mask_size_mm": scene.grid * args.pitch * 1e3,
            "focal_length_m": args.focal_length,
            "detector_pitch_um": det_pitch * 1e6,
            "detector_fov_mm": fov_mm,
            "target_width_mm": z_width * 1e3,
        }

    np.save(out / "target.npy", scene.canvas)
    np.save(out / "aperture.npy", aperture)
    np.save(out / "pattern_intensity.npy", intensity)
    np.save(out / "pattern_amplitude.npy", np.sqrt(intensity))
    save_aperture_png(out / "aperture.png", aperture, scene.modulation)

    r0, r1, c0, c1 = scene.roi
    roi_peak = float(intensity[r0:r1, c0:c1].max())
    roi_display = display(intensity[r0:r1, c0:c1], roi_peak)
    Image.fromarray((display(intensity, roi_peak) * 255).astype(np.uint8)).save(out / "pattern_fit.png")
    # ROI 裁剪：和 comparison.png ② 是同一块，112×112，可直接当“相机只拍这一块”的图
    Image.fromarray((roi_display * 255).astype(np.uint8)).save(out / "pattern_roi.png")
    Image.fromarray((scene.target * 255).astype(np.uint8)).save(out / "target_roi.png")
    np.save(out / "history.npy", np.array([best["history"]["step"], best["history"]["data"],
                                           best["history"]["outside"]]))

    tag = f"{scene.source}   →   {scene.modulation} 光阑  ({scene.grid}×{scene.grid})"
    render(out / "comparison.png", scene, aperture, intensity, metrics, best["history"], tag)

    result = {
        "target": scene.source,
        "modulation": scene.modulation,
        "grid": scene.grid,
        "roi": metrics["roi"],
        "roi_box": {
            "rows": [r0, r1], "cols": [c0, c1],
            "size": [r1 - r0, c1 - c0],
            "target_rows": [r0 + args.margin, r0 + args.margin + args.target_size],
            "target_cols": [c0 + args.margin, c0 + args.margin + args.target_size],
        },
        "metrics": {k: v for k, v in metrics.items() if k not in ("roi",)},
        "warnings": (["掩模退化成近似全黑/全空，本次结果不可用：换一张对比度更高的图，"
                      "或加大 --steps / --target-size"] if metrics["degenerate"] else []),
        "geometry": geometry,
        "design": {
            "steps": args.steps,
            "restarts": args.restarts,
            "seed": args.seed,
            "best_score": best["score"],
            "runs": all_runs,
        },
        "physics_note": (
            "实振幅光阑的远场强度共轭对称：I(k)=I(-k)，ROI 内的目标会在中心对称位置"
            "产生一个旋转 180° 的孪生像（corr_twin≈1）。要得到单个目标，只取 ROI；"
            "或者改用 phase 调制（复振幅，无此限制）。"
            if scene.modulation != "phase" else
            "相位光阑为复振幅，无共轭对称限制，目标位于光轴中心且没有孪生像。"),
        "command": "python design.py " + " ".join(sys.argv[1:]),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n完成：{out}")
    print(f"  corr_amplitude = {metrics['corr_amplitude']:.4f}   "
          f"MAE = {metrics['mae_amplitude']:.4f}   "
          f"ROI 能量占比 = {metrics['roi_energy_fraction']*100:.2f}%")
    if metrics["degenerate"]:
        print("  [警告] 掩模退化成近似全黑/全空，这次结果不可用；"
              "换对比度更高的图，或加大 --steps / --target-size 再试")
    if geometry:
        print(f"  掩模 {geometry['mask_size_mm']:.2f} mm，f = {args.focal_length} m 时 "
              f"单像素 {geometry['detector_pitch_um']:.1f} um，视场 {geometry['detector_fov_mm']:.1f} mm，"
              f"目标宽约 {geometry['target_width_mm']:.1f} mm")


if __name__ == "__main__":
    main()
