"""单张衍射图样 -> 光阑：冻结正向模型 + 测试时优化，不训练逆网络。

整个逆向流程只依赖 generate.py 和 train.py 里的正向定义：
    - generate.PhysicsConfig / group_name / postprocess
    - train.ForwardModel / FFT / ASM 层

文件内部按顺序分为四块：
    1. 正向算子：checkpoint、FrozenForward、自洽 z、图样读写
    2. 几何后处理：连通域清理、椭圆/矩形拟合
    3. 反演核心：配置、物理先验、PixelField、多尺度/多起点、top-3
    4. 输出与 CLI：画图、保存、单图推理入口

用法：
    python inverse.py \
        --checkpoint runs/v1/nf_1/physics/best_model.pt \
        --pattern path/to/pattern.npy \
        --nf 1 \
        --out inverse_runs/case_nf1

    离线生成测试用例（只写 pattern/gt，不反演）：
    python inverse.py --write-cases inverse_runs/cases --nf 1 --case-scale 0.5

损失调度（默认 --loss-schedule phased）：
    - [0, fit)          只拟合数据（L1 + 相关系数 + log 频谱 + 低频大方向），形状先验恒为 0；
    - [fit, prior)      平滑（TV/Laplacian）、面积、连通性先验按各自窗口渐入；
    - [prior, detail)   二值、开运算、小岛先验渐入，配合 STE 与更小的平滑尺度收细节；
    - 先验总权重按 EMA 自适应，保持“先验 / 空间 L1”的固定占比（默认 0.25）。
    --loss-schedule fixed 可复现旧的固定权重行为。

配对光阑仅用于对比（不参与网络输入）：
    --expected / --expected-sample 给出参考光阑，运行结束后生成
    expected_vs_top1.png 和诊断数字；网络输入始终只有 --pattern。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate import (PhysicsConfig, angular_spectrum_raw, fraunhofer_raw,
                      geometry, group_name, postprocess)
from train import (AngularSpectrumDetectorLayer, ForwardModel,
                   FraunhoferDetectorLayer, choose_device)


# ============================================================
# 1. 正向算子
# ============================================================
# ============================================================
# 1.1 checkpoint 解析与加载
# ============================================================
def resolve_checkpoint(run: Optional[str] = None, nf: Optional[float] = None,
                       model: str = "physics",
                       checkpoint: Optional[str] = None) -> Path:
    """解析 best_model.pt。

    优先使用 --checkpoint；否则从 --run/<nf 组>/<model>/best_model.pt 查找。
    组名与 generate.group_name 一致，例如 nf=0.1 -> nf_0p1。
    """
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint 不存在：{path}")
        return path
    if not run:
        raise ValueError("必须提供 --checkpoint 或 --run")
    root = Path(run).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"run 目录不存在：{root}")
    if nf is None:
        raise ValueError("从 --run 查找 checkpoint 时必须提供 --nf")
    path = root / group_name(float(nf)) / model / "best_model.pt"
    if path.is_file():
        return path
    available = sorted(root.glob(f"nf_*/{model}/best_model.pt"))
    names = [p.parent.parent.name for p in available]
    raise FileNotFoundError(
        f"找不到 {path}；当前可用的组：{', '.join(names) if names else '无'}")


def load_forward_checkpoint(path: str | Path, device: torch.device):
    """加载并冻结 physics checkpoint，返回 (checkpoint, config, model)。"""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "physics":
        raise ValueError(
            f"只支持 physics checkpoint，当前为 {checkpoint.get('kind')!r}；"
            "baseline 没有物理传播层，不能用于反演")
    config = PhysicsConfig(**checkpoint["physics_config"])
    model = ForwardModel(
        checkpoint["kind"],
        checkpoint["base_ch"],
        config.log_alpha,
        propagator=checkpoint["propagator"],
        physics_config=config,
        asm_precision=checkpoint.get("asm_precision", "float32"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return checkpoint, config, model


# ============================================================
# 1.2 冻结正向算子
# ============================================================
class FrozenForward(torch.nn.Module):
    """可微但参数冻结的正向算子。

    mode="checkpoint"：
        使用完整 physics 网络（传播层 + 残差 U-Net），和正向比较结果一致。
        z 只进入 condition（当前 checkpoint 的传播层是 FFT）。
    mode="physical"：
        只用纯 FFT / ASM，按 NF 与 far_threshold 自动选择，不含学习参数。
    """

    def __init__(self, checkpoint: str | Path, mode: str = "checkpoint",
                 device: str | torch.device = "cuda",
                 residual_weight: float = 1.0,
                 nf: Optional[float] = None):
        super().__init__()
        if mode not in ("checkpoint", "physical"):
            raise ValueError("mode 必须是 checkpoint 或 physical")
        self.device = torch.device(device)
        self.checkpoint_path = str(Path(checkpoint).resolve())
        ckpt, config, model = load_forward_checkpoint(checkpoint, self.device)
        self.config = config
        checkpoint_nf = float(ckpt["nf"])
        if nf is not None and abs(float(nf) - checkpoint_nf) > 1e-9:
            raise ValueError(
                f"指定的 NF={nf:g} 与 checkpoint 的 NF={checkpoint_nf:g} 不一致")
        self.nf = checkpoint_nf
        self.mode = mode
        self.residual_weight = float(residual_weight)
        self.asm_precision = ckpt.get("asm_precision", "float32")
        self.condition_mean = torch.tensor(
            ckpt["condition_mean"], dtype=torch.float32, device=self.device)
        self.condition_std = torch.tensor(
            ckpt["condition_std"], dtype=torch.float32, device=self.device)

        if mode == "checkpoint":
            self.model = model
            self.physical_layer = None
            self._check_layer = None
        else:
            # 纯物理模式不需要学习参数；保留 model 会浪费显存，直接丢弃。
            self.model = None
            self.physical_layer = self._build_physical_layer()
            self._check_layer = self.physical_layer

        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _build_physical_layer(self) -> torch.nn.Module:
        if self.nf < self.config.far_threshold:
            layer = FraunhoferDetectorLayer(self.config.log_alpha)
        else:
            layer = AngularSpectrumDetectorLayer(
                self.config, precision=self.asm_precision)
        return layer.to(self.device).eval()

    # ---------- 几何 ----------
    def condition_from_z(self, z_m: torch.Tensor) -> torch.Tensor:
        """把以米为单位的 z 转成 checkpoint 的标准化 condition。"""
        z = torch.as_tensor(z_m, dtype=torch.float64, device=self.device).reshape(-1)
        z = z.clamp_min(1e-12)
        detector_pitch = (self.config.wavelength_m * z /
                          (self.config.grid * self.config.input_pitch_m))
        values = torch.stack((torch.log(z), torch.log(detector_pitch)), dim=1)
        values = values.to(torch.float32)
        return (values - self.condition_mean) / self.condition_std

    @staticmethod
    def support_width_batch(mask: torch.Tensor, pitch: float) -> torch.Tensor:
        """batch 版支撑宽度 D（米）；空 mask 返回 0。"""
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask 必须是 [B,1,H,W]")
        batch, _, height, width = mask.shape
        rows = mask.sum(dim=3) > 0          # [B,1,H]
        cols = mask.sum(dim=2) > 0          # [B,1,W]
        ih = torch.arange(height, device=mask.device).view(1, 1, height)
        iw = torch.arange(width, device=mask.device).view(1, 1, width)
        row_min = torch.where(rows, ih, torch.full_like(ih, height)).amin(-1)
        row_max = torch.where(rows, ih, torch.full_like(ih, -1)).amax(-1)
        col_min = torch.where(cols, iw, torch.full_like(iw, width)).amin(-1)
        col_max = torch.where(cols, iw, torch.full_like(iw, -1)).amax(-1)
        span_row = (row_max - row_min + 1).clamp_min(0)
        span_col = (col_max - col_min + 1).clamp_min(0)
        span = torch.maximum(span_row, span_col).squeeze(1)
        return span * float(pitch)

    def geometry_from_mask(self, mask: torch.Tensor,
                           nf: Optional[float] = None,
                           support_fraction: float = 0.5):
        """由当前 mask 自洽计算 D、z、condition。

        mask 应该是 detach 后的二值/近似二值张量；梯度不通过支撑宽度回传。
        """
        nf = float(self.nf if nf is None else nf)
        if nf <= 0:
            raise ValueError("NF 必须为正")
        support = self.support_width_batch(mask, self.config.input_pitch_m)
        fallback = (float(support_fraction) * self.config.grid *
                    self.config.input_pitch_m)
        support = torch.where(
            support > 0, support,
            torch.full_like(support, fallback))
        z = support.square() / (self.config.wavelength_m * nf)
        z = z.clamp_min(1e-9)
        condition = self.condition_from_z(z)
        return support, z, condition

    # ---------- 前向 ----------
    def forward(self, aperture: torch.Tensor,
                z_m: Optional[torch.Tensor] = None,
                condition: Optional[torch.Tensor] = None,
                return_aux: bool = False):
        """aperture: [B,1,H,W] -> log 图样 [B,1,H,W]。

        return_aux=True 时返回 (pred, sim, delta)。physical 模式下 delta=0。
        """
        if z_m is None:
            raise ValueError("必须提供 z_m；可用 geometry_from_mask 自洽计算")
        z = torch.as_tensor(z_m, dtype=torch.float64, device=aperture.device).reshape(-1)
        if z.numel() == 1 and aperture.shape[0] > 1:
            z = z.expand(aperture.shape[0])
        if z.numel() != aperture.shape[0]:
            raise ValueError("z 的数量必须与 batch 一致")
        if condition is None:
            condition = self.condition_from_z(z)
        condition = condition.to(device=aperture.device, dtype=torch.float32)

        if self.mode == "checkpoint":
            pred, sim, delta = self.model(aperture, condition, z)
            pred = sim + self.residual_weight * delta
        else:
            pred = self.physical_layer(aperture, z)
            sim = pred
            delta = torch.zeros_like(pred)

        if return_aux:
            return pred, sim, delta
        return pred

    @torch.no_grad()
    def physical_pattern(self, aperture: torch.Tensor,
                         z_m: torch.Tensor) -> torch.Tensor:
        """用纯 FFT/ASM 做独立物理检查，不受残差网络影响。"""
        if self._check_layer is None:
            self._check_layer = self._build_physical_layer()
        return self._check_layer(aperture, z_m)

    def condition_zscore(self, condition: torch.Tensor) -> torch.Tensor:
        return condition.detach().abs().amax(dim=-1)


# ============================================================
# 1.3 图样读写
# ============================================================
def load_pattern(path: str | Path, grid: int,
                 input_mode: str = "dataset",
                 invert: bool = False,
                 resize: bool = True) -> torch.Tensor:
    """读取单张衍射图样，返回 [1,1,grid,grid] float32。

    input_mode="dataset"：按 data/v1 的 [0,1] log 图样处理，只做 clip。
    input_mode="image"：按原始强度图处理，做 peak 归一化 + log1p(alpha*I)。
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"图样不存在：{path}")

    if path.suffix.lower() == ".npy":
        array = np.load(path)
        array = np.squeeze(array)
        if array.ndim == 3:
            if array.shape[0] in (3, 4):
                array = array[..., :3].mean(axis=0)
            else:
                array = array[0]
        if array.ndim != 2:
            raise ValueError(f"不支持 .npy 形状：{np.load(path).shape}")
        array = np.asarray(array, dtype=np.float32)
        if input_mode == "image":
            array = postprocess(array)
        else:
            array = np.clip(array, 0.0, 1.0)
    else:
        image = Image.open(path).convert("L")
        array = np.asarray(image, dtype=np.float32) / 255.0
        if invert:
            array = 1.0 - array
        if input_mode == "image":
            array = postprocess(array)
        else:
            array = np.clip(array, 0.0, 1.0)

    if not np.isfinite(array).all():
        raise ValueError(f"图样包含非有限值：{path}")
    if float(array.max()) <= 0:
        raise ValueError(f"图样全为零：{path}")

    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    if tensor.shape[-2:] != (grid, grid):
        if not resize:
            raise ValueError(f"图样尺寸 {tuple(tensor.shape[-2:])} 与 grid={grid} 不一致")
        tensor = F.interpolate(tensor, size=(grid, grid), mode="bilinear",
                               align_corners=False)
    return tensor.clamp(0.0, 1.0)


def _as_aperture_tensor(array: np.ndarray, grid: int) -> torch.Tensor:
    """把参考光阑转成 [1,1,grid,grid] 二值 mask。"""
    array = np.array(array, dtype=np.float32, copy=True)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"预期光阑必须是 2D，当前形状 {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("预期光阑包含非有限值")
    peak = float(array.max())
    if peak > 1.0 + 1e-6:
        array = array / peak
    tensor = torch.from_numpy(array)[None, None]
    if tensor.shape[-2:] != (grid, grid):
        tensor = F.interpolate(tensor, size=(grid, grid), mode="bilinear",
                               align_corners=False)
    mask = (tensor > 0.5).float()
    if float(mask.sum()) <= 0:
        raise ValueError("预期光阑为空（没有 >0.5 的像素）")
    return mask


def load_reference_aperture(path: str | Path, grid: int) -> torch.Tensor:
    """读取“预期光阑”（.npy / .png）；只用于对比，不进入优化。"""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"预期光阑不存在：{path}")
    if path.suffix.lower() == ".npy":
        array = np.squeeze(np.load(path))
        return _as_aperture_tensor(array, grid)
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return _as_aperture_tensor(array, grid)


def load_reference_sample(dataset: str | Path, index: int,
                          grid: int) -> torch.Tensor:
    """从 data/*/apertures.npy 取指定样本作为预期光阑（只用于对比）。"""
    root = Path(dataset).expanduser()
    path = root / "apertures.npy"
    if not path.is_file():
        raise FileNotFoundError(f"找不到 apertures.npy：{path}")
    apertures = np.load(path, mmap_mode="r")
    if not 0 <= index < len(apertures):
        raise ValueError(f"样本编号 {index} 超出范围 [0,{len(apertures) - 1}]")
    array = np.asarray(apertures[index], dtype=np.float32)
    return _as_aperture_tensor(array, grid)


# ============================================================
# 1.4 测试用例生成（离线，只写 pattern/gt，不参与反演）
# ============================================================
def _case_canvas(size, supersample=4):
    scale = max(int(supersample), 1)
    n = size * scale
    image = Image.new("L", (n, n), 0)
    return image, ImageDraw.Draw(image), n


def _case_finish(image, size):
    array = np.asarray(image.resize((size, size), Image.BILINEAR),
                       dtype=np.float32) / 255.0
    aperture = (array > 0.5).astype(np.float32)
    if not aperture.any():
        raise ValueError("光阑为空")
    return aperture


def _case_digit_zero(size=256, supersample=4, scale=1.0):
    image, draw, n = _case_canvas(size, supersample)
    cx, cy = n / 2, n / 2
    rx, ry, thick = 0.235 * n * scale, 0.315 * n * scale, 0.055 * n * scale
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    draw.ellipse([cx - rx + thick, cy - ry + thick,
                  cx + rx - thick, cy + ry - thick], fill=0)
    return _case_finish(image, size)


def _case_digit_eight(size=256, supersample=4, scale=1.0):
    image, draw, n = _case_canvas(size, supersample)
    cx, cy = n / 2, n / 2
    half_w, half_h = 0.215 * n * scale, 0.165 * n * scale
    thick = 0.05 * n * scale
    for offset in (-0.155 * n * scale, 0.155 * n * scale):
        draw.ellipse([cx - half_w, cy + offset - half_h,
                      cx + half_w, cy + offset + half_h], fill=255)
    for offset in (-0.155 * n * scale, 0.155 * n * scale):
        draw.ellipse([cx - half_w + thick, cy + offset - half_h + thick,
                      cx + half_w - thick, cy + offset + half_h - thick], fill=0)
    return _case_finish(image, size)


def _case_disk(size=256, supersample=4, scale=1.0):
    image, draw, n = _case_canvas(size, supersample)
    cx, cy, radius = n / 2, n / 2, 0.24 * n * scale
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)
    return _case_finish(image, size)


def _case_polygon(sides):
    def make(size=256, supersample=4, scale=1.0):
        image, draw, n = _case_canvas(size, supersample)
        cx, cy, radius = n / 2, n / 2, 0.25 * n * scale
        points = []
        for k in range(sides):
            angle = 2.0 * np.pi * k / sides - np.pi / 2.0
            points.append((cx + radius * np.cos(angle),
                           cy + radius * np.sin(angle)))
        draw.polygon(points, fill=255)
        return _case_finish(image, size)
    return make


CASE_BUILDERS = {
    "digit0": _case_digit_zero,
    "digit8": _case_digit_eight,
    "disk": _case_disk,
    "triangle": _case_polygon(3),
    "square": _case_polygon(4),
    "pentagon": _case_polygon(5),
    "hexagon": _case_polygon(6),
    "octagon": _case_polygon(8),
}


def write_cases(out: str | Path, shapes, nf: float = 1.0, grid: int = 256,
                scale: float = 0.5):
    """生成测试用例：gt.npy / pattern.npy / meta.json。

    这是离线数据准备；反演只读 pattern.npy，gt.npy 仅用于 --expected 对比。
    """
    config = PhysicsConfig(grid=grid)
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    suffix = f"_nf{nf:g}" + ("" if scale == 1.0 else f"_s{scale:g}")
    for name in shapes:
        if name not in CASE_BUILDERS:
            raise ValueError(f"未知用例：{name}")
        aperture = CASE_BUILDERS[name](grid, scale=scale)
        geo = geometry(aperture, nf, config)
        raw = (fraunhofer_raw(aperture, geo["z_m"], config)
               if nf < config.far_threshold
               else angular_spectrum_raw(aperture, geo["z_m"], config))
        pattern = postprocess(raw, config.log_alpha)
        folder = root / f"{name}{suffix}"
        folder.mkdir(exist_ok=True)
        np.save(folder / "gt.npy", aperture)
        np.save(folder / "pattern.npy", pattern.astype(np.float32))
        write_json(folder / "meta.json", {
            "name": name, "nf": nf, "grid": grid, "scale": scale,
            "z_m": geo["z_m"], "D_m": geo["D_m"],
            "detector_pitch_m": geo["detector_pitch_m"],
            "area_fraction": float(aperture.mean()),
            "note": "gt.npy/pattern.npy 只用于离线用例；inverse 只读 pattern.npy",
        })
        print(f"{name}: {folder}  z={geo['z_m']:.4f} m  "
              f"area={aperture.mean():.3f}")


# ============================================================
# 2. 几何后处理
# ============================================================
@dataclass
class Component:
    rows: np.ndarray
    cols: np.ndarray
    area: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    major: float
    minor: float
    extent_major: float
    extent_minor: float
    angle: float
    solidity: float


def _components(mask: np.ndarray, min_area: int,
                max_components: int) -> List[Component]:
    """4-邻域连通域标记；返回按面积从大到小排序的组件。"""
    mask = np.asarray(mask > 0.5, dtype=bool)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    result: List[Component] = []
    for start_r in range(height):
        for start_c in range(width):
            if not mask[start_r, start_c] or visited[start_r, start_c]:
                continue
            queue = deque([(start_r, start_c)])
            visited[start_r, start_c] = True
            rows, cols = [], []
            while queue:
                r, c = queue.popleft()
                rows.append(r)
                cols.append(c)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < height and 0 <= nc < width and
                            mask[nr, nc] and not visited[nr, nc]):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            rows_arr = np.asarray(rows, dtype=np.float32)
            cols_arr = np.asarray(cols, dtype=np.float32)
            area = len(rows)
            if area < min_area:
                continue
            r0, r1 = int(rows_arr.min()), int(rows_arr.max())
            c0, c1 = int(cols_arr.min()), int(cols_arr.max())
            bbox_area = max((r1 - r0 + 1) * (c1 - c0 + 1), 1)
            solidity = area / bbox_area
            cr, cc = float(rows_arr.mean()), float(cols_arr.mean())
            dr, dc = rows_arr - cr, cols_arr - cc
            cov_rr = float((dr * dr).mean())
            cov_cc = float((dc * dc).mean())
            cov_rc = float((dr * dc).mean())
            # 协方差矩阵特征值 -> 主轴长度；均匀椭圆 var=r^2/4。
            trace = cov_rr + cov_cc
            diff = cov_rr - cov_cc
            root = float(np.sqrt(max(diff * diff + 4.0 * cov_rc * cov_rc, 0.0)))
            lambda_major = max((trace + root) / 2.0, 1e-4)
            lambda_minor = max((trace - root) / 2.0, 1e-4)
            # 均匀填充椭圆沿主轴的方差 = 半轴^2/4；因此直径 = 4*sqrt(var)。
            major = 4.0 * float(np.sqrt(lambda_major))
            minor = 4.0 * float(np.sqrt(lambda_minor))
            angle = 0.5 * float(np.arctan2(2.0 * cov_rc, diff))
            ca, sa = float(np.cos(angle)), float(np.sin(angle))
            proj_major = dr * ca + dc * sa
            proj_minor = -dr * sa + dc * ca
            extent_major = float(proj_major.max() - proj_major.min() + 1.0)
            extent_minor = float(proj_minor.max() - proj_minor.min() + 1.0)
            result.append(Component(
                rows=rows_arr, cols=cols_arr, area=area,
                centroid=(cr, cc), bbox=(r0, r1, c0, c1),
                major=major, minor=minor,
                extent_major=extent_major, extent_minor=extent_minor,
                angle=angle, solidity=solidity))
    result.sort(key=lambda item: item.area, reverse=True)
    return result[:max_components]


def _ellipse_mask(shape: Tuple[int, int], component: Component,
                  shrink: float = 0.92) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    ca, sa = np.cos(component.angle), np.sin(component.angle)
    xr = (xx - component.centroid[1]) * ca - (yy - component.centroid[0]) * sa
    yr = (xx - component.centroid[1]) * sa + (yy - component.centroid[0]) * ca
    rx = max(1.2, component.major * 0.5 * shrink)
    ry = max(1.2, component.minor * 0.5 * shrink)
    return ((xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0).astype(np.float32)


def _rectangle_mask(shape: Tuple[int, int], component: Component,
                    shrink: float = 1.0) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    ca, sa = np.cos(component.angle), np.sin(component.angle)
    xr = (xx - component.centroid[1]) * ca - (yy - component.centroid[0]) * sa
    yr = (xx - component.centroid[1]) * sa + (yy - component.centroid[0]) * ca
    half_w = max(1.0, component.extent_major * 0.5 * shrink)
    half_h = max(1.0, component.extent_minor * 0.5 * shrink)
    return ((np.abs(xr) <= half_w) & (np.abs(yr) <= half_h)).astype(np.float32)


def remove_small_components(mask: np.ndarray, min_area: int = 12,
                            max_components: int = 64) -> np.ndarray:
    """只保留面积 >= min_area 的连通域（输出平滑前先去散点）。"""
    mask = np.asarray(mask > 0.5, dtype=np.float32)
    output = np.zeros_like(mask)
    for component in _components(mask, min_area=1,
                                 max_components=max_components):
        if component.area >= min_area:
            output[component.rows.astype(int),
                   component.cols.astype(int)] = 1.0
    return output


def clean_aperture(mask: np.ndarray, mode: str = "ellipse",
                   min_area: int = 12, max_fit_area: int = 1200,
                   min_solidity: float = 0.5,
                   max_components: int = 64,
                   shrink: float = 0.92) -> Tuple[np.ndarray, dict]:
    """几何整理；返回 (clean_mask, info)。

    mode="none" 直接返回原图；"ellipse" 或 "rect" 只拟合中小型且较实的组件。
    """
    mask = np.asarray(mask > 0.5, dtype=np.float32)
    if mode == "none":
        return mask, {"components_in": 0, "components_fit": 0,
                      "components_removed": 0, "components_kept": 0}

    components = _components(mask, min_area=1, max_components=max_components)
    output = np.zeros_like(mask)
    fit_count = removed = kept = 0
    for component in components:
        if component.area < min_area:
            removed += 1
            continue
        can_fit = (component.area <= max_fit_area and
                   component.solidity >= min_solidity)
        if can_fit:
            if mode == "rect":
                primitive = _rectangle_mask(mask.shape, component, shrink=1.0)
            else:
                primitive = _ellipse_mask(mask.shape, component, shrink=shrink)
            # 拟合结果太少或太大时回退到原始像素，避免把小孔拟合成一个点。
            if primitive.sum() < max(6, 0.25 * component.area) or \
                    primitive.sum() > 2.5 * component.area:
                output[component.rows.astype(int),
                       component.cols.astype(int)] = 1.0
                kept += 1
            else:
                output = np.maximum(output, primitive)
                fit_count += 1
        else:
            output[component.rows.astype(int),
                   component.cols.astype(int)] = 1.0
            kept += 1
    return output.astype(np.float32), {
        "components_in": len(components),
        "components_fit": fit_count,
        "components_removed": removed,
        "components_kept": kept,
        "mode": mode,
    }


# ============================================================
# 3. 反演核心
# ============================================================
# ============================================================
# 3.1 配置与结果
# ============================================================
@dataclass
class InversionConfig:
    nf: float
    grid: int = 256
    scales: Tuple[int, ...] = (64, 128, 256)
    steps: Tuple[int, ...] = (240, 240, 480)
    lrs: Tuple[float, ...] = (0.05, 0.02, 0.01)
    restarts: int = 4
    top_k: int = 3
    geometry: str = "self-consistent"   # self-consistent | canonical | known
    support_fraction: float = 0.5
    z_m: Optional[float] = None
    geometry_every: int = 10

    temperature_start: float = 1.0
    temperature_end: float = 0.35
    ste_steps: int = 160

    # 损失调度：前期只拟合大方向，后期逐项加入并加强形状约束。
    loss_schedule: str = "phased"        # phased | fixed
    fit_fraction: float = 0.30           # [0, fit)：只拟合数据
    prior_fraction: float = 0.60         # [fit, prior)：平滑/面积/连通性渐入
    detail_fraction: float = 0.85        # [prior, detail)：二值/形态学渐入
    prior_ratio: float = 0.25            # 先验总权重 / 空间 L1 的目标占比
    prior_gain_min: float = 0.0
    prior_gain_max: float = 50.0
    ema_decay: float = 0.90
    lambda_coarse: float = 0.25          # 前期低频数据项（大方向）权重
    coarse_kernel: int = 9
    sigma_coarse_scale: float = 1.6      # 粗阶段平滑尺度（相对 smooth_sigma）
    sigma_detail_scale: float = 1.0      # 细节阶段平滑尺度（不锐化，保持平滑边缘）
    edge_sigma: float = 1.5              # 结束后的边缘平滑（像素），0=关闭

    # 后段几何参数化：椭圆/正多边形基元并集 − 孔洞，参数少、边界天然平滑。
    geometric_stage: bool = True
    geom_primitive: str = "ellipse"      # ellipse | polygon
    geom_sides: int = 6                  # polygon 模式的边数
    geom_max_positive: int = 8
    geom_max_negative: int = 4
    geom_steps: int = 300
    geom_lr: float = 0.02
    geom_softness: float = 25.0
    geom_min_area: int = 20
    geom_support_weight: float = 0.01
    geom_area_weight: float = 0.005

    lambda_freq: float = 0.02
    lambda_corr: float = 0.5
    lambda_tv: float = 1e-3
    lambda_lap: float = 3e-3
    lambda_perim: float = 1e-3
    lambda_bin: float = 1e-3
    lambda_open: float = 1e-3
    lambda_open5: float = 0.0
    lambda_island: float = 3e-2
    lambda_area: float = 1e-2
    lambda_area_center: float = 0.1
    lambda_res: float = 1e-3
    area_low: float = 0.002
    area_high: float = 0.6
    smooth_sigma: float = 1.0
    island_kernel: int = 7
    island_min_mass: float = 6.0
    primitive_fit: str = "none"          # none | ellipse | rect
    primitive_min_area: int = 20
    primitive_max_fit_area: int = 4000
    primitive_min_solidity: float = 0.2
    primitive_max_components: int = 32
    primitive_shrink: float = 1.0

    seed: int = 42
    verbose: bool = True

    def validate(self):
        if len(self.scales) != len(self.steps) or len(self.scales) != len(self.lrs):
            raise ValueError("scales/steps/lrs 长度必须一致")
        if self.grid < 32 or self.grid % 8:
            raise ValueError("grid 必须 >=32 且能被 8 整除")
        if self.nf <= 0:
            raise ValueError("NF 必须为正")
        if self.restarts < 1 or self.top_k < 1:
            raise ValueError("restarts/top_k 必须为正")
        if self.geometry not in ("self-consistent", "canonical", "known"):
            raise ValueError("未知 geometry 模式")
        if self.geometry == "known" and self.z_m is None:
            raise ValueError("known 模式需要 z_m")
        if self.support_fraction <= 0 or self.support_fraction > 1:
            raise ValueError("support_fraction 必须在 (0,1] 内")
        if self.geometry_every < 1:
            raise ValueError("geometry_every 必须 >=1")
        if self.smooth_sigma < 0:
            raise ValueError("smooth_sigma 不能为负")
        if min(self.lambda_tv, self.lambda_lap, self.lambda_perim,
               self.lambda_bin, self.lambda_open, self.lambda_open5,
               self.lambda_island, self.lambda_area,
               self.lambda_area_center, self.lambda_res,
               self.lambda_freq, self.lambda_corr) < 0:
            raise ValueError("损失权重不能为负")
        if self.island_kernel < 3 or self.island_kernel % 2 == 0:
            raise ValueError("island_kernel 必须是 >=3 的奇数")
        if self.loss_schedule not in ("phased", "fixed"):
            raise ValueError("loss_schedule 必须是 phased 或 fixed")
        if not 0.0 <= self.fit_fraction < self.prior_fraction <= self.detail_fraction <= 1.0:
            raise ValueError("必须满足 0 <= fit_fraction < prior_fraction "
                             "<= detail_fraction <= 1")
        if not 0.0 <= self.prior_gain_min <= self.prior_gain_max:
            raise ValueError("prior_gain_min/max 必须非负且 min <= max")
        if self.prior_ratio < 0:
            raise ValueError("prior_ratio 不能为负")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay 必须在 [0,1) 内")
        if self.lambda_coarse < 0:
            raise ValueError("lambda_coarse 不能为负")
        if self.coarse_kernel < 3 or self.coarse_kernel % 2 == 0:
            raise ValueError("coarse_kernel 必须是 >=3 的奇数")
        if self.sigma_coarse_scale <= 0 or self.sigma_detail_scale <= 0:
            raise ValueError("sigma 缩放系数必须为正")
        if self.edge_sigma < 0:
            raise ValueError("edge_sigma 不能为负")
        if not 0.0 < self.temperature_end <= 2.0:
            raise ValueError("temperature_end 必须在 (0,2] 内")
        if self.geom_steps < 0:
            raise ValueError("geom_steps 不能为负")
        if self.geom_lr <= 0 or self.geom_softness <= 0:
            raise ValueError("geom_lr / geom_softness 必须为正")
        if self.geom_max_positive < 1 or self.geom_max_negative < 0:
            raise ValueError("geom 基元数量不合法")
        if self.geom_min_area < 1:
            raise ValueError("geom_min_area 必须 >=1")
        if self.geom_support_weight < 0 or self.geom_area_weight < 0:
            raise ValueError("geom 正则权重不能为负")
        if self.geom_primitive not in ("ellipse", "polygon"):
            raise ValueError("geom_primitive 必须是 ellipse 或 polygon")
        if self.geom_sides < 3:
            raise ValueError("geom_sides 必须 >=3")
        if self.primitive_fit not in ("none", "ellipse", "rect"):
            raise ValueError("primitive_fit 必须是 none/ellipse/rect")


@dataclass
class Candidate:
    rank: int
    seed: int
    init: str
    rank_score: float
    data_loss: float
    prior_loss: float
    total_loss: float
    pattern_mae: float
    physical_mae: float
    residual_ratio: float
    area_fraction: float
    support_fraction: float
    z_m: float
    condition_zscore: float
    aperture_soft: np.ndarray
    aperture_bin: np.ndarray
    pattern_pred: np.ndarray
    pattern_physical: np.ndarray
    history: List[Dict[str, float]]
    runtime_s: float
    pairwise_iou: Dict[int, float] = field(default_factory=dict)
    primitive_components: int = 0
    primitive_info: Dict = field(default_factory=dict)
    loss_gain: float = 1.0
    coarse_loss: float = 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "rank": self.rank,
            "seed": self.seed,
            "init": self.init,
            "rank_score": self.rank_score,
            "data_loss": self.data_loss,
            "prior_loss": self.prior_loss,
            "total_loss": self.total_loss,
            "pattern_mae": self.pattern_mae,
            "physical_mae": self.physical_mae,
            "residual_ratio": self.residual_ratio,
            "area_fraction": self.area_fraction,
            "support_fraction": self.support_fraction,
            "z_m": self.z_m,
            "condition_zscore": self.condition_zscore,
            "runtime_s": self.runtime_s,
            "primitive_components": self.primitive_components,
            "primitive_info": self.primitive_info,
            "loss_gain": self.loss_gain,
            "coarse_loss": self.coarse_loss,
        }


# ============================================================
# 3.2 可微物理先验
# ============================================================
def correlation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """尺度不变的相关系数损失；避免“空光阑/大光斑”这类平凡解占便宜。"""
    p = pred.flatten(1)
    t = target.flatten(1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    numerator = (p * t).sum(dim=1)
    denominator = (p.square().sum(dim=1).sqrt() *
                   t.square().sum(dim=1).sqrt()).clamp_min(1e-12)
    return (1.0 - numerator / denominator).mean()


def pattern_loss(pred: torch.Tensor, target: torch.Tensor,
                 lambda_freq: float = 0.02,
                 lambda_corr: float = 0.5):
    """L1 + 相关系数 + log 频谱 L1。"""
    l1 = F.l1_loss(pred, target)
    corr = (correlation_loss(pred, target) if lambda_corr > 0
            else pred.new_zeros(()))
    if lambda_freq > 0:
        f_pred = torch.fft.fft2(pred[:, 0])
        f_target = torch.fft.fft2(target[:, 0])
        freq = F.l1_loss(torch.log1p(f_pred.abs()),
                         torch.log1p(f_target.abs()))
    else:
        freq = pred.new_zeros(())
    total = l1 + lambda_freq * freq + lambda_corr * corr
    return total, l1, freq, corr


def tv_loss(aperture: torch.Tensor) -> torch.Tensor:
    """各向同性 TV，近似边界周长。"""
    dx = aperture[..., :, 1:] - aperture[..., :, :-1]
    dy = aperture[..., 1:, :] - aperture[..., :-1, :]
    # dx: B,1,H,W-1；dy: B,1,H-1,W；裁到公共的 H-1,W-1 后逐像素合成。
    dx = dx[..., :-1, :]
    dy = dy[..., :, :-1]
    return torch.sqrt(dx.square() + dy.square() + 1e-6).mean()


def laplacian_loss(aperture: torch.Tensor) -> torch.Tensor:
    """边界曲率/锯齿惩罚：二阶差分绝对值和。

    细丝、尖刺和棋盘格会产生大的 Laplacian；平滑边界只在边缘有稀疏响应。
    """
    lap = (aperture[..., 2:, 1:-1] + aperture[..., :-2, 1:-1] +
           aperture[..., 1:-1, 2:] + aperture[..., 1:-1, :-2] -
           4.0 * aperture[..., 1:-1, 1:-1])
    return lap.abs().mean()


def perimeter_ratio(aperture: torch.Tensor) -> torch.Tensor:
    """周长/面积比：扭曲、细长、破碎的形状比值高。

    圆形最低；光栅/字母会偏高，所以只作为弱先验，权重不能大。
    这里直接用 TV 的均值除以面积均值，避免乘 N² 后量级失控。
    """
    area = aperture.mean(dim=(1, 2, 3)).clamp_min(1e-4)
    return (tv_loss(aperture) / area).mean()


def island_loss(aperture: torch.Tensor, kernel: int = 7,
                min_mass: float = 6.0) -> torch.Tensor:
    """孤立小岛惩罚：局部邻域内前景质量太低的像素。

    真实的孔/缝在 7x7 邻域内会有足够多的前景像素；随机孤立点、细碎毛刺
    的局部质量接近 1。用 avg_pool 的局部和做可微近似，避免连通域标记。
    """
    pad = kernel // 2
    local_mass = F.avg_pool2d(aperture, kernel, stride=1, padding=pad)
    local_mass = local_mass * float(kernel * kernel)
    penalty = F.relu(1.0 - local_mass / max(float(min_mass), 1e-3))
    numerator = (aperture * penalty).sum(dim=(1, 2, 3))
    denominator = aperture.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    return (numerator / denominator).mean()


def binary_loss(aperture: torch.Tensor) -> torch.Tensor:
    """4a(1-a)，在 [0,1] 上归一化到 0..1。"""
    return (4.0 * aperture * (1.0 - aperture)).mean()


def soft_erode(aperture: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    pad = kernel // 2
    return -F.max_pool2d(-aperture, kernel, stride=1, padding=pad)


def soft_dilate(aperture: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    pad = kernel // 2
    return F.max_pool2d(aperture, kernel, stride=1, padding=pad)


def opening_loss(aperture: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """软形态学开运算惩罚：去掉小于 kernel 的细碎结构。

    对二值图，opening(a) <= a；a - opening(a) 正好是被开运算移除的像素。
    对 soft mask，用 relu 保持非负，避免鼓励 a > opening(a)。
    """
    opened = soft_dilate(soft_erode(aperture, kernel), kernel)
    return F.relu(aperture - opened).mean()


def area_hinge(aperture: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """只惩罚极端面积，不约束具体形状面积。"""
    area = aperture.mean(dim=(1, 2, 3))
    return (F.relu(low - area).square() + F.relu(area - high).square()).mean()


def coarse_pattern_loss(pred: torch.Tensor, target: torch.Tensor,
                        kernel: int = 9) -> torch.Tensor:
    """低频数据项：低通后的 L1，在大方向阶段帮助先定轮廓。"""
    kernel = max(int(kernel), 3)
    if kernel % 2 == 0:
        kernel += 1
    low_pred = F.avg_pool2d(pred, kernel, stride=1, padding=kernel // 2)
    low_target = F.avg_pool2d(target, kernel, stride=1, padding=kernel // 2)
    return F.l1_loss(low_pred, low_target)


def smoothstep(value: float) -> float:
    """0..1 的平滑过渡。"""
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _ramp(progress: float, start: float, end: float) -> float:
    if end <= start + 1e-9:
        return 1.0 if progress >= end else 0.0
    return smoothstep((float(progress) - start) / (end - start))


def loss_factors(progress: float, cfg: InversionConfig) -> Dict[str, float]:
    """各损失项在当前进度的相对权重（0..1）。

    phased（默认）：
        [0, fit)        粗项 1→0，形状先验全 0：只拟合数据和大方向；
        [fit, prior)    平滑/面积先验先到满权重，连通性（perim/island）稍后；
        [prior, detail) 二值/形态学先验收细节；
        [detail, 1]     全部满权重，交给 STE 与更小的 smooth_sigma。
    fixed：复现旧行为（二值项 0.2→1，开运算 15% 后渐入）。
    """
    progress = float(np.clip(progress, 0.0, 1.0))
    if cfg.loss_schedule == "fixed":
        opening = max(0.0, (progress - 0.15) / 0.85)
        return {"coarse": 0.0, "tv": 1.0, "lap": 1.0, "perim": 1.0,
                "island": 1.0, "area": 1.0, "area_center": 1.0,
                "bin": 0.2 + 0.8 * progress, "open": opening,
                "open5": opening}
    f1, f2, f3 = cfg.fit_fraction, cfg.prior_fraction, cfg.detail_fraction
    span = max(f2 - f1, 1e-6)
    smoothing = _ramp(progress, f1, f1 + 0.60 * span)
    structure = _ramp(progress, f1 + 0.35 * span, f2)
    detail = _ramp(progress, f2, f3)
    coarse = 1.0 - _ramp(progress, f1, f1 + 0.50 * span)
    # 面积护栏（只惩罚 <0.2% 或 >60% 的退化面积）全程生效，
    # 防止“无形状先验”的前期把光阑整块压没或撑满。
    return {"coarse": coarse, "tv": smoothing, "lap": smoothing,
            "perim": structure, "island": structure,
            "area": 1.0, "area_center": smoothing,
            "bin": detail, "open": detail, "open5": detail}


def prior_loss(aperture: torch.Tensor, cfg: InversionConfig,
               progress: float, area_target: Optional[float] = None,
               factors: Optional[Dict[str, float]] = None):
    """形状先验：返回 (按当前因子加权的先验, 统计)。

    总权重再由调用方乘以自适应 gain（见 run_restart），
    这样“先验 / 空间 L1”的占比在训练中可以动态保持。
    """
    factors = factors or loss_factors(progress, cfg)
    tv = tv_loss(aperture)
    lap = laplacian_loss(aperture)
    perimeter = perimeter_ratio(aperture)
    binary = binary_loss(aperture)
    opening = opening_loss(aperture, kernel=3)
    opening5 = opening_loss(aperture, kernel=5)
    island = island_loss(aperture, kernel=cfg.island_kernel,
                         min_mass=cfg.island_min_mass)
    area = area_hinge(aperture, cfg.area_low, cfg.area_high)
    terms = {
        "tv": cfg.lambda_tv * tv,
        "lap": cfg.lambda_lap * lap,
        "perim": cfg.lambda_perim * perimeter,
        "bin": cfg.lambda_bin * binary,
        "open": cfg.lambda_open * opening,
        "open5": cfg.lambda_open5 * opening5,
        "island": cfg.lambda_island * island,
        "area": cfg.lambda_area * area,
    }
    if cfg.lambda_area_center > 0 and area_target is not None:
        area_value = aperture.mean(dim=(1, 2, 3)).clamp_min(1e-6)
        area_center = (torch.log(area_value) -
                       math.log(max(float(area_target), 1e-6))).square().mean()
        terms["area_center"] = cfg.lambda_area_center * area_center
    else:
        area_center = aperture.new_zeros(())
        terms["area_center"] = area_center
    raw = sum(terms.values())
    ramped = sum(terms[key] * factors[key] for key in terms)
    stats = {key: float(value.detach()) for key, value in terms.items()}
    stats.update({
        "prior_raw": float(raw.detach()),
        "prior_ramped": float(ramped.detach()),
        "f_tv": float(factors["tv"]),
        "f_perim": float(factors["perim"]),
        "f_bin": float(factors["bin"]),
        "f_coarse": float(factors["coarse"]),
    })
    return ramped, stats


# ============================================================
# 3.3 光阑参数化与初始化
# ============================================================
class PixelField(nn.Module):
    """平滑水平集参数化：sigmoid(GaussianBlur(theta)/T)。

    GaussianBlur 把边界限制在有限曲率/有限分辨率内，减少细丝、尖刺和棋盘格；
    它不是训练层，kernel 固定，只作为物理上的“最小特征尺寸/边缘平滑”先验。
    """

    def __init__(self, theta: torch.Tensor, smooth_sigma: float = 1.5):
        super().__init__()
        self.theta = nn.Parameter(theta)
        self.temperature = 1.0
        self.register_buffer("blur_kernel", torch.empty(0, device=theta.device),
                             persistent=False)
        self._build_kernel(float(smooth_sigma))

    def _build_kernel(self, sigma: float):
        """重建固定高斯核；sigma 随训练阶段变化时调用，不产生梯度。"""
        sigma = float(max(sigma, 0.0))
        self.smooth_sigma = sigma
        self.blur_radius = max(1, int(math.ceil(3.0 * sigma))) if sigma > 0 else 0
        if self.blur_radius == 0:
            self.blur_kernel = torch.empty(0, device=self.theta.device)
            return
        coords = torch.arange(-self.blur_radius, self.blur_radius + 1,
                              dtype=torch.float32, device=self.theta.device)
        kernel = torch.exp(-coords.square() / (2.0 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = (kernel[:, None] * kernel[None, :])[None, None]
        self.blur_kernel = kernel.repeat(self.theta.shape[1], 1, 1, 1)

    def set_sigma(self, sigma: float):
        """训练中切换平滑尺度：粗阶段更平滑，细节阶段更锐利。"""
        if abs(float(sigma) - self.smooth_sigma) > 1e-6:
            self._build_kernel(float(sigma))

    def forward(self, size: int) -> torch.Tensor:
        logits = self.theta
        if self.blur_radius > 0:
            logits = F.conv2d(logits, self.blur_kernel,
                              padding=self.blur_radius,
                              groups=logits.shape[1])
        aperture = torch.sigmoid(logits /
                                 max(float(self.temperature), 1e-3))
        if aperture.shape[-1] != size:
            aperture = F.interpolate(aperture, size=(size, size),
                                     mode="bilinear", align_corners=False)
        return aperture.clamp(1e-6, 1.0 - 1e-6)


def resize_theta(theta: torch.Tensor, size: int) -> torch.Tensor:
    if theta.shape[-1] == size:
        return theta
    return F.interpolate(theta, size=(size, size), mode="bilinear",
                         align_corners=False)


def blur_mask(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """对输出光阑做高斯平滑，得到平滑边缘；不参与梯度、不训练。"""
    if sigma <= 0:
        return mask
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=mask.dtype,
                          device=mask.device)
    kernel = torch.exp(-coords.square() / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None]
    kernel = kernel.repeat(mask.shape[1], 1, 1, 1)
    padded = F.pad(mask, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel, groups=mask.shape[1])


class PrimitiveField(torch.nn.Module):
    """后段几何参数化：基元并集 −（孔洞并集），渲染成平滑光阑。

    primitive="ellipse"：每个基元 (cx, cy, rx, ry, angle)；
    primitive="polygon"：每个基元 (cx, cy, R, angle)，所有基元共用一个边数 sides。
    内部场 >0，用 max 求并集、sigmoid 生成平滑边界。
    """

    def __init__(self, positives, negatives=None, softness: float = 25.0,
                 primitive: str = "ellipse", sides: int = 6):
        super().__init__()
        self.softness = float(softness)
        self.primitive = primitive
        self.sides = int(sides)
        self.pos_raw = nn.Parameter(
            torch.as_tensor(np.asarray(positives, np.float32)))
        if negatives is not None and len(negatives):
            self.neg_raw = nn.Parameter(
                torch.as_tensor(np.asarray(negatives, np.float32)))
        else:
            self.neg_raw = None

    @staticmethod
    def _ellipse_fields(params: torch.Tensor, grid: int) -> torch.Tensor:
        device, dtype = params.device, params.dtype
        cx = params[:, 0].view(-1, 1, 1).clamp(0.0, grid - 1.0)
        cy = params[:, 1].view(-1, 1, 1).clamp(0.0, grid - 1.0)
        rx = (F.softplus(params[:, 2]) + 0.8).view(-1, 1, 1).clamp(
            2.0, 0.45 * grid)
        ry = (F.softplus(params[:, 3]) + 0.8).view(-1, 1, 1).clamp(
            2.0, 0.45 * grid)
        angle = params[:, 4].view(-1, 1, 1)
        yy, xx = torch.meshgrid(
            torch.arange(grid, device=device, dtype=dtype),
            torch.arange(grid, device=device, dtype=dtype),
            indexing="ij")
        x = xx[None] - cx
        y = yy[None] - cy
        ca, sa = torch.cos(angle), torch.sin(angle)
        xr = ca * x + sa * y
        yr = -sa * x + ca * y
        return 1.0 - (xr / rx).square() - (yr / ry).square()

    @staticmethod
    def _polygon_fields(params: torch.Tensor, grid: int,
                        sides: int) -> torch.Tensor:
        device, dtype = params.device, params.dtype
        cx = params[:, 0].view(-1, 1, 1).clamp(0.0, grid - 1.0)
        cy = params[:, 1].view(-1, 1, 1).clamp(0.0, grid - 1.0)
        radius = (F.softplus(params[:, 2]) + 1.0).view(-1, 1, 1).clamp(
            2.0, 0.5 * grid)
        angle = params[:, 3].view(-1, 1, 1)
        yy, xx = torch.meshgrid(
            torch.arange(grid, device=device, dtype=dtype),
            torch.arange(grid, device=device, dtype=dtype),
            indexing="ij")
        x = xx[None] - cx
        y = yy[None] - cy
        ca, sa = torch.cos(angle), torch.sin(angle)
        xr = ca * x + sa * y
        yr = -sa * x + ca * y
        normals = (2.0 * math.pi / sides) * torch.arange(
            sides, device=device, dtype=dtype) + math.pi / sides
        projection = (xr[None] * torch.cos(normals).view(-1, 1, 1, 1) +
                      yr[None] * torch.sin(normals).view(-1, 1, 1, 1))
        apothem = (radius * math.cos(math.pi / sides)).clamp_min(1e-3)
        return 1.0 - projection.amax(dim=0) / apothem

    def _fields(self, params: torch.Tensor, grid: int) -> torch.Tensor:
        if self.primitive == "polygon":
            return self._polygon_fields(params, grid, self.sides)
        return self._ellipse_fields(params, grid)

    def forward(self, grid: int) -> torch.Tensor:
        field = self._fields(self.pos_raw, grid).amax(dim=0)
        if self.neg_raw is not None:
            field = field - self._fields(self.neg_raw, grid).amax(dim=0)
        field = field.clamp(-1.0, 1.0)
        return torch.sigmoid(self.softness * field)[None, None]


def primitives_from_mask(mask: np.ndarray, max_positive: int = 8,
                         max_negative: int = 4, min_area: int = 20,
                         primitive: str = "ellipse", sides: int = 6):
    """从像素解提取基元：连通域 → 正基元，内部孔洞 → 负基元。"""
    binary = np.asarray(mask > 0.5, dtype=np.float32)

    def params_of(component):
        if primitive == "polygon":
            area = max(float(component.area), 1.0)
            radius = math.sqrt(area / (0.5 * sides *
                                       math.sin(2.0 * math.pi / sides)))
            return [component.centroid[1], component.centroid[0],
                    max(radius, 1.0), 0.0]
        return [component.centroid[1], component.centroid[0],
                max(component.major * 0.5, 1.0),
                max(component.minor * 0.5, 1.0),
                component.angle]

    positives = [params_of(component) for component in
                 _components(binary, min_area=min_area,
                             max_components=max_positive)]
    if not positives:
        center = (binary.shape[0] - 1) / 2.0
        radius = max(6.0, 0.12 * binary.shape[0])
        positives = ([[center, center, radius, radius, 0.0]]
                     if primitive == "ellipse"
                     else [[center, center, radius, 0.0]])

    height, width = binary.shape
    negatives = []
    complement = (1.0 - binary).astype(np.float32)
    for component in _components(complement, min_area=max(min_area, 6),
                                 max_components=max_negative * 4):
        r0, r1, c0, c1 = component.bbox
        if r0 <= 1 or c0 <= 1 or r1 >= height - 2 or c1 >= width - 2:
            continue  # 外部背景，不是孔
        negatives.append(params_of(component))
        if len(negatives) >= max_negative:
            break
    return positives, negatives


def refine_geometry(observed: torch.Tensor, forward: "FrozenForward",
                    cfg: "InversionConfig", init_mask: torch.Tensor,
                    z: Optional[torch.Tensor],
                    condition: Optional[torch.Tensor]):
    """后段：把像素解拟合成平滑基元（椭圆/正多边形），再用同一前向微调。"""
    device = observed.device
    positives, negatives = primitives_from_mask(
        init_mask[0, 0].detach().cpu().numpy(),
        cfg.geom_max_positive, cfg.geom_max_negative, cfg.geom_min_area,
        cfg.geom_primitive, cfg.geom_sides)
    field = PrimitiveField(positives, negatives, cfg.geom_softness,
                           cfg.geom_primitive, cfg.geom_sides).to(device)
    optimizer = torch.optim.Adam(field.parameters(), lr=cfg.geom_lr)
    init = init_mask.detach().float()
    support = F.max_pool2d((init > 0.5).float(), kernel_size=11,
                           stride=1, padding=5)
    area_target = float(init.mean())
    with torch.enable_grad():
        for step in range(cfg.geom_steps):
            aperture = field(cfg.grid)
            if (cfg.geometry == "self-consistent" and
                    step % cfg.geometry_every == 0):
                _, z, condition = _update_geometry(forward, cfg, aperture, z)
            pred = forward(aperture, z, condition)
            data, _, _, _ = pattern_loss(pred, observed, cfg.lambda_freq,
                                         cfg.lambda_corr)
            outside = ((aperture * (1.0 - support)).sum() /
                       aperture.sum().clamp_min(1e-6))
            area = aperture.mean().clamp_min(1e-6)
            area_term = (torch.log(area) -
                         math.log(max(area_target, 1e-6))).square()
            loss = (data + cfg.geom_support_weight * outside +
                    cfg.geom_area_weight * area_term)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(field.parameters(), 1.0)
            optimizer.step()
    with torch.no_grad():
        soft = field(cfg.grid).clamp(1e-6, 1.0 - 1e-6)
        binary = (soft > 0.5).float()
    info = {
        "geometric_stage": True,
        "primitive": cfg.geom_primitive,
        "sides": cfg.geom_sides if cfg.geom_primitive == "polygon" else None,
        "positive": len(positives),
        "negative": len(negatives),
        "positive_primitives": field.pos_raw.detach().cpu().tolist(),
        "negative_primitives": (field.neg_raw.detach().cpu().tolist()
                                if field.neg_raw is not None else []),
    }
    return soft, binary, info


def _initial_mask(size: int, kind: str, rng: np.random.Generator,
                  device: torch.device, area_fraction: float) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(size, device=device, dtype=torch.float32),
        torch.arange(size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    area_fraction = float(np.clip(area_fraction, 0.002, 0.6))
    if kind == "blob":
        cx = size / 2 + rng.uniform(-0.12, 0.12) * size
        cy = size / 2 + rng.uniform(-0.12, 0.12) * size
        aspect = rng.uniform(0.55, 1.45)
        radius = math.sqrt(area_fraction / (math.pi * aspect)) * size
        rx, ry = radius * math.sqrt(aspect), radius / math.sqrt(aspect)
        angle = rng.uniform(0, math.pi)
        ca, sa = math.cos(angle), math.sin(angle)
        xr = (xx - cx) * ca - (yy - cy) * sa
        yr = (xx - cx) * sa + (yy - cy) * ca
        mask = ((xr / max(rx, 1.0)) ** 2 +
                (yr / max(ry, 1.0)) ** 2 <= 1.0).float()
    elif kind == "ring":
        cx = size / 2 + rng.uniform(-0.08, 0.08) * size
        cy = size / 2 + rng.uniform(-0.08, 0.08) * size
        outer = size * rng.uniform(0.18, 0.42)
        inner = outer * rng.uniform(0.55, 0.85)
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        mask = ((d2 <= outer ** 2) & (d2 >= inner ** 2)).float()
    elif kind == "grating":
        angle = rng.uniform(0, math.pi)
        period = size * rng.uniform(0.025, 0.09)
        duty = rng.uniform(0.25, 0.65)
        cx = size / 2 + rng.uniform(-0.08, 0.08) * size
        cy = size / 2 + rng.uniform(-0.08, 0.08) * size
        half_w = size * rng.uniform(0.22, 0.42)
        half_h = size * rng.uniform(0.22, 0.42)
        ca, sa = math.cos(angle), math.sin(angle)
        xr = (xx - cx) * ca - (yy - cy) * sa
        yr = (xx - cx) * sa + (yy - cy) * ca
        inside = (xr.abs() <= half_w) & (yr.abs() <= half_h)
        mask = (inside & ((xr % period) / period < duty)).float()
    else:
        # sparse / random：低分辨率随机场阈值化，面积由 area_fraction 控制。
        low_res = 24 if kind == "random" else 48
        low = torch.from_numpy(
            rng.normal(size=(1, 1, low_res, low_res)).astype(np.float32)).to(device)
        kind_mode = "bilinear" if kind == "random" else "nearest"
        smooth = F.interpolate(low, size=(size, size), mode=kind_mode,
                               align_corners=False if kind_mode == "bilinear" else None)
        threshold = torch.quantile(smooth.flatten(), 1.0 - area_fraction)
        mask = (smooth >= threshold).float()
    return mask


def init_theta(size: int, kind: str, seed: int,
               device: torch.device,
               area_fraction: float = 0.1) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    mask = _initial_mask(size, kind, rng, device, area_fraction)
    aperture = 0.05 + 0.90 * mask
    aperture = aperture + torch.from_numpy(
        rng.normal(scale=0.03, size=(1, 1, size, size)).astype(np.float32)).to(device)
    aperture = aperture.clamp(0.02, 0.98)
    return torch.logit(aperture)


def restart_plan(restarts: int, seed: int):
    """多起点计划：覆盖不同面积和不同几何类型，避免单一 blob 局部极小。"""
    rng = np.random.default_rng(seed)
    base = [
        ("sparse", 0.02), ("random", 0.05), ("blob", 0.03),
        ("grating", 0.08), ("ring", 0.04), ("sparse", 0.01),
        ("random", 0.10), ("blob", 0.015), ("grating", 0.04),
        ("ring", 0.08), ("sparse", 0.05), ("random", 0.02),
    ]
    plan = []
    for index in range(restarts):
        if index < len(base):
            kind, area = base[index]
        else:
            kind = str(rng.choice(["sparse", "random", "blob", "grating", "ring"]))
            area = float(rng.uniform(0.01, 0.2))
        # 轻微扰动面积，避免多个 restart 完全一样。
        area = float(np.clip(area * rng.uniform(0.75, 1.25), 0.005, 0.3))
        plan.append((kind, area))
    return plan


# ============================================================
# 3.4 单次 restart
# ============================================================
def _fixed_geometry(forward: FrozenForward, cfg: InversionConfig,
                    device: torch.device):
    """canonical / known 模式的固定 z 与 condition。"""
    if cfg.geometry == "known":
        z = torch.tensor([float(cfg.z_m)], dtype=torch.float64, device=device)
    else:
        reference = (cfg.support_fraction * cfg.grid *
                     forward.config.input_pitch_m)
        z_value = reference ** 2 / (forward.config.wavelength_m * cfg.nf)
        z = torch.tensor([z_value], dtype=torch.float64, device=device)
    return z, forward.condition_from_z(z)


def _update_geometry(forward: FrozenForward, cfg: InversionConfig,
                     aperture: torch.Tensor,
                     z_previous: Optional[torch.Tensor]):
    """self-consistent 模式：用当前 mask 的支撑更新 z。"""
    mask = (aperture.detach() > 0.5).float()
    support, z_new, condition = forward.geometry_from_mask(
        mask, nf=cfg.nf, support_fraction=cfg.support_fraction)
    if z_previous is not None:
        # 轻微 EMA，避免支撑在优化中跳变时 condition 抖动。
        z_new = 0.7 * z_previous.to(z_new.dtype) + 0.3 * z_new
        condition = forward.condition_from_z(z_new)
    return support, z_new.detach(), condition.detach()


def run_restart(observed: torch.Tensor, forward: FrozenForward,
                cfg: InversionConfig, seed: int, init_kind: str,
                area_fraction: float) -> Candidate:
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    started = time.perf_counter()
    device = observed.device

    theta = init_theta(cfg.scales[0], init_kind, seed, device,
                       area_fraction=area_fraction)
    field = PixelField(theta, smooth_sigma=cfg.smooth_sigma).to(device)
    history: List[Dict[str, float]] = []

    z: Optional[torch.Tensor] = None
    condition: Optional[torch.Tensor] = None
    if cfg.geometry != "self-consistent":
        z, condition = _fixed_geometry(forward, cfg, device)

    total_steps = max(sum(cfg.steps), 1)
    global_step = 0
    stage_count = len(cfg.scales)
    ema_l1: Optional[float] = None
    ema_prior: Optional[float] = None
    gain = 1.0
    target_ratio = 0.0

    for stage, (resolution, steps, lr) in enumerate(
            zip(cfg.scales, cfg.steps, cfg.lrs)):
        if stage > 0:
            # 不要原地改 Parameter.data 的形状，否则 autograd 仍记着旧形状。
            field = PixelField(
                resize_theta(field.theta.detach(), resolution),
                smooth_sigma=cfg.smooth_sigma).to(device)
        # 平滑尺度：粗阶段更大（先定大方向），最后一级回落到基础值。
        # fixed 模式保持旧的固定 sigma，便于对照。
        if cfg.loss_schedule == "phased":
            stage_fraction = stage / max(stage_count - 1, 1)
            sigma_scale = (cfg.sigma_coarse_scale * (1.0 - stage_fraction) +
                           1.0 * stage_fraction)
            field.set_sigma(max(0.2, cfg.smooth_sigma * sigma_scale))
        optimizer = torch.optim.Adam([field.theta], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(steps, 1))
        final_stage = stage == stage_count - 1
        ste_start = max(0, steps - min(cfg.ste_steps, steps))

        for step in range(steps):
            progress = (global_step + step) / max(total_steps - 1, 1)
            field.temperature = (cfg.temperature_start +
                                 (cfg.temperature_end - cfg.temperature_start)
                                 * progress)
            # 细节阶段：更小的平滑尺度 + 后续 STE，用于收紧具体形状。
            if (cfg.loss_schedule == "phased" and final_stage and
                    progress >= cfg.detail_fraction):
                field.set_sigma(max(0.2, cfg.smooth_sigma *
                                    cfg.sigma_detail_scale))
            aperture = field(cfg.grid)
            factors = loss_factors(progress, cfg)

            if cfg.geometry == "self-consistent":
                if z is None or step % cfg.geometry_every == 0:
                    _, z, condition = _update_geometry(
                        forward, cfg, aperture, z)
            elif z is None:
                z, condition = _fixed_geometry(forward, cfg, device)

            use_ste = final_stage and step >= ste_start
            if cfg.loss_schedule == "phased":
                use_ste = use_ste and progress >= cfg.detail_fraction
            if use_ste:
                hard = (aperture > 0.5).float()
                aperture_used = aperture + (hard - aperture).detach()
            else:
                aperture_used = aperture

            pred, simulated, delta = forward(
                aperture_used, z, condition, return_aux=True)
            data, l1, freq, corr = pattern_loss(
                pred, observed, cfg.lambda_freq, cfg.lambda_corr)
            if cfg.lambda_coarse > 0 and factors["coarse"] > 0:
                coarse = coarse_pattern_loss(pred, observed, cfg.coarse_kernel)
            else:
                coarse = pred.new_zeros(())
            coarse_contrib = cfg.lambda_coarse * factors["coarse"] * coarse

            prior_ramped, prior_stats = prior_loss(
                aperture, cfg, progress, area_target=area_fraction,
                factors=factors)
            l1_value = float(l1.detach())
            raw_value = float(prior_stats["prior_raw"])
            decay = float(cfg.ema_decay)
            if ema_l1 is None:
                ema_l1, ema_prior = l1_value, raw_value
            else:
                ema_l1 = decay * ema_l1 + (1.0 - decay) * l1_value
                ema_prior = decay * ema_prior + (1.0 - decay) * raw_value
            if cfg.loss_schedule == "phased":
                target_ratio = cfg.prior_ratio * _ramp(
                    progress, cfg.fit_fraction, cfg.prior_fraction)
                gain = target_ratio * ema_l1 / max(ema_prior, 1e-8)
                gain = float(np.clip(gain, cfg.prior_gain_min,
                                     cfg.prior_gain_max))
            else:
                target_ratio, gain = 0.0, 1.0
            prior = gain * prior_ramped
            residual = delta.abs().mean()
            loss = data + coarse_contrib + prior + cfg.lambda_res * residual

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([field.theta], 1.0)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                # STE 的硬阈值梯度偶尔会把个别 logit 推到 NaN/Inf；
                # 这里做有限性保护，避免 final soft mask 出现 NaN。
                field.theta.data = torch.nan_to_num(
                    field.theta.data, nan=0.0,
                    posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

            if step % 10 == 0 or (final_stage and step == steps - 1):
                history.append({
                    "step": global_step + step,
                    "stage": stage,
                    "resolution": resolution,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "loss": float(loss.detach()),
                    "data": float(data.detach()),
                    "l1": float(l1.detach()),
                    "freq": float(freq.detach()),
                    "corr": float(corr.detach()),
                    "coarse": float(coarse.detach()),
                    "coarse_contrib": float(coarse_contrib.detach()),
                    "prior": float(prior.detach()),
                    "prior_raw": raw_value,
                    "prior_ramped": float(prior_ramped.detach()),
                    "gain": float(gain),
                    "target_ratio": float(target_ratio),
                    "prior_ratio": float(prior.detach()) / max(l1_value, 1e-12),
                    "sigma": float(field.smooth_sigma),
                    "phase": (0 if progress < cfg.fit_fraction else
                              1 if progress < cfg.detail_fraction else 2),
                    "tv": prior_stats["tv"],
                    "lap": prior_stats["lap"],
                    "perim": prior_stats["perim"],
                    "bin": prior_stats["bin"],
                    "open": prior_stats["open"],
                    "open5": prior_stats["open5"],
                    "island": prior_stats["island"],
                    "area": prior_stats["area"],
                    "f_tv": prior_stats["f_tv"],
                    "f_perim": prior_stats["f_perim"],
                    "f_bin": prior_stats["f_bin"],
                    "f_coarse": prior_stats["f_coarse"],
                    "residual": float(residual.detach()),
                    "z_m": float(z.detach().reshape(-1)[0]),
                    "area_fraction": float((aperture > 0.5).float().mean()),
                })
        global_step += steps

    with torch.no_grad():
        aperture_soft_raw = torch.nan_to_num(field(cfg.grid), nan=0.0,
                                             posinf=1.0, neginf=0.0)
        aperture_bin = (aperture_soft_raw > 0.5).float()
        aperture_soft = aperture_soft_raw
        primitive_info: Dict = {}
        primitive_components = 0
        if cfg.geometric_stage:
            geo_soft, geo_bin, primitive_info = refine_geometry(
                observed, forward, cfg, aperture_bin, z, condition)
            aperture_soft, aperture_bin = geo_soft, geo_bin
            primitive_components = int(
                primitive_info.get("positive", 0) +
                primitive_info.get("negative", 0))
        else:
            if cfg.primitive_fit != "none":
                cleaned, primitive_info = clean_aperture(
                    aperture_bin[0, 0].cpu().numpy(),
                    mode=cfg.primitive_fit,
                    min_area=cfg.primitive_min_area,
                    max_fit_area=cfg.primitive_max_fit_area,
                    min_solidity=cfg.primitive_min_solidity,
                    max_components=cfg.primitive_max_components,
                    shrink=cfg.primitive_shrink,
                )
                if cleaned.any():
                    aperture_bin = torch.from_numpy(
                        cleaned)[None, None].to(device)
                    aperture_soft = aperture_bin
                    primitive_components = int(
                        primitive_info.get("components_fit", 0) +
                        primitive_info.get("components_kept", 0))
                else:
                    primitive_info["fallback"] = "kept_raw_pixel_mask"
                    primitive_components = int(
                        primitive_info.get("components_in", 0))
            # 去散点 + 边缘平滑：对二值 mask 做高斯平滑后再取 0.5 等值线。
            cleaned = remove_small_components(
                aperture_bin[0, 0].cpu().numpy(),
                min_area=max(cfg.primitive_min_area, 6),
                max_components=cfg.primitive_max_components)
            if cleaned.any():
                aperture_bin = torch.from_numpy(cleaned)[None, None].to(device)
            if cfg.edge_sigma > 0:
                aperture_soft = blur_mask(aperture_bin,
                                          cfg.edge_sigma).clamp(1e-6, 1.0 - 1e-6)
                aperture_bin = (aperture_soft > 0.5).float()
            else:
                aperture_soft = aperture_bin
        support, z_self, condition_self = forward.geometry_from_mask(
            aperture_bin, nf=cfg.nf, support_fraction=cfg.support_fraction)
        if cfg.geometry == "self-consistent":
            z_final, condition_final = z_self, condition_self
        else:
            # known / canonical 模式下优化用的是固定 z，最终评估也必须用同一个。
            z_final, condition_final = _fixed_geometry(forward, cfg, device)
        pred_final, simulated_final, delta_final = forward(
            aperture_soft, z_final, condition_final, return_aux=True)
        physical_final = forward.physical_pattern(aperture_soft, z_final)
        final_progress = 1.0
        final_prior_ramped, _ = prior_loss(
            aperture_soft, cfg, final_progress,
            area_target=float(aperture_bin.mean()))
        final_prior = float(gain) * float(final_prior_ramped.detach())
        final_data, _, _, _ = pattern_loss(
            pred_final, observed, cfg.lambda_freq, cfg.lambda_corr)
        final_coarse = float(coarse_pattern_loss(pred_final, observed,
                                                 cfg.coarse_kernel))
        pattern_mae = float((pred_final - observed).abs().mean())
        physical_mae = float((physical_final - observed).abs().mean())
        residual_ratio = float(
            delta_final.abs().mean() /
            simulated_final.abs().mean().clamp_min(1e-12))
        area_fraction = float(aperture_bin.mean())
        support_fraction = float(
            support.reshape(-1)[0] /
            (cfg.grid * forward.config.input_pitch_m))
        condition_zscore = float(
            forward.condition_zscore(condition_final).mean())
        rank_score = float(final_data + 0.05 * final_prior)

    return Candidate(
        rank=-1,
        seed=seed,
        init=f"{init_kind}@{area_fraction:.3f}",
        rank_score=rank_score,
        data_loss=float(final_data),
        prior_loss=float(final_prior),
        total_loss=float(final_data + final_prior),
        pattern_mae=pattern_mae,
        physical_mae=physical_mae,
        residual_ratio=residual_ratio,
        area_fraction=area_fraction,
        support_fraction=support_fraction,
        z_m=float(z_final.reshape(-1)[0]),
        condition_zscore=condition_zscore,
        aperture_soft=aperture_soft[0, 0].detach().cpu().numpy(),
        aperture_bin=aperture_bin[0, 0].detach().cpu().numpy(),
        pattern_pred=pred_final[0, 0].detach().cpu().numpy(),
        pattern_physical=physical_final[0, 0].detach().cpu().numpy(),
        history=history,
        runtime_s=time.perf_counter() - started,
        primitive_components=primitive_components,
        primitive_info=primitive_info,
        loss_gain=float(gain),
        coarse_loss=final_coarse,
    )


# ============================================================
# 3.5 top-k 多样性选择
# ============================================================
def _shift_to_center(mask: torch.Tensor) -> torch.Tensor:
    rows, cols = torch.nonzero(mask[0, 0] > 0.5, as_tuple=True)
    if rows.numel() == 0:
        return mask
    height, width = mask.shape[-2:]
    shift_r = int(round(height / 2.0 - float(rows.float().mean())))
    shift_c = int(round(width / 2.0 - float(cols.float().mean())))
    return torch.roll(mask, shifts=(shift_r, shift_c), dims=(-2, -1))


def centroid_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """质心对齐后的 IoU，用来判断两个候选是不是同一个形状。"""
    a = torch.from_numpy((mask_a > 0.5).astype(np.float32))[None, None]
    b = torch.from_numpy((mask_b > 0.5).astype(np.float32))[None, None]
    a = _shift_to_center(a)
    b = _shift_to_center(b)
    inter = (a * b).sum()
    union = ((a + b) > 0).float().sum()
    return float(inter / union.clamp_min(1.0))


def select_diverse(candidates: Sequence[Candidate], k: int = 3,
                   iou_threshold: float = 0.8) -> List[Candidate]:
    """按 rank_score 排序，贪心选择形状不同的 top-k。

    先过滤几乎全空的候选；它们虽然可能多样性高，但物理上不是有效光阑。
    如果过滤后不足 k 个，再退回全部候选。
    """
    valid = [candidate for candidate in candidates
             if candidate.area_fraction >= 1e-3]
    pool = valid if len(valid) >= k else list(candidates)
    ordered = sorted(pool, key=lambda item: item.rank_score)
    selected: List[Candidate] = []
    for candidate in ordered:
        if all(centroid_iou(candidate.aperture_bin, other.aperture_bin)
               < iou_threshold for other in selected):
            selected.append(candidate)
        if len(selected) >= k:
            break
    if len(selected) < k:
        for candidate in ordered:
            if all(candidate is not chosen for chosen in selected):
                selected.append(candidate)
            if len(selected) >= k:
                break
    for rank, candidate in enumerate(selected, start=1):
        candidate.rank = rank
        candidate.pairwise_iou = {
            other.rank: centroid_iou(candidate.aperture_bin, other.aperture_bin)
            for other in selected if other is not candidate
        }
    return selected


def invert(observed: torch.Tensor, forward: FrozenForward,
           cfg: InversionConfig):
    """运行 R 个 restart，返回 (top-k 候选, 全部候选)。"""
    cfg.validate()
    observed = observed.to(forward.device)
    candidates: List[Candidate] = []
    plan = restart_plan(cfg.restarts, cfg.seed)
    for restart, (kind, area_fraction) in enumerate(plan):
        seed = cfg.seed + restart * 1009
        if cfg.verbose:
            print(f"  restart {restart + 1}/{cfg.restarts}: "
                  f"init={kind} area~{area_fraction:.3f} seed={seed}",
                  flush=True)
        candidate = run_restart(observed, forward, cfg, seed, kind,
                                area_fraction)
        candidates.append(candidate)
        if cfg.verbose:
            print(f"    data={candidate.data_loss:.3e} "
                  f"pattern_mae={candidate.pattern_mae:.3e} "
                  f"area={candidate.area_fraction:.3f} "
                  f"z={candidate.z_m:.3f} "
                  f"({candidate.runtime_s:.1f}s)", flush=True)
    return select_diverse(candidates, cfg.top_k), candidates


# ============================================================
# 4. 输出与 CLI
# ============================================================
# ============================================================
# 4.1 保存与画图
# ============================================================
def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.floating):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload: Dict):
    with path.open("w", encoding="utf-8") as file:
        json.dump(_jsonable(payload), file, ensure_ascii=False, indent=2)


def write_history(path: Path, history: List[Dict]):
    if not history:
        return
    keys = list(dict.fromkeys(key for row in history for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def save_image(path: Path, array: np.ndarray, cmap: str = "gray",
               vmin: Optional[float] = None, vmax: Optional[float] = None):
    plt.imsave(path, np.asarray(array), cmap=cmap, vmin=vmin, vmax=vmax)


def plot_loss(history: List[Dict], path: Path, title: str):
    if not history:
        return
    steps = [row["step"] for row in history]
    fig, (ax, ax_weight) = plt.subplots(
        2, 1, figsize=(7.2, 6.0), sharex=True,
        gridspec_kw={"height_ratios": (3.0, 2.0)})
    for key, label in (("loss", "total"), ("data", "data"),
                       ("prior", "prior"), ("coarse_contrib", "coarse")):
        values = [max(float(row.get(key, 0.0)), 1e-12) for row in history]
        ax.semilogy(steps, values, label=label, lw=1.3)
    ax.set_ylabel("loss (floor 1e-12)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    target = [float(row.get("target_ratio", 0.0)) for row in history]
    actual = [float(row.get("prior_ratio", 0.0)) for row in history]
    ax_weight.plot(steps, target, "k--", lw=1.2, label="target prior / L1")
    ax_weight.plot(steps, actual, color="C3", lw=1.4,
                   label="actual prior / L1")
    ax_weight.set(xlabel="step", ylabel="prior / L1")
    ax_weight.grid(alpha=0.25)
    twin = ax_weight.twinx()
    twin.plot(steps, [float(row.get("gain", 1.0)) for row in history],
              color="C2", lw=1.0, alpha=0.7, label="adaptive gain")
    twin.set_ylabel("gain")
    handles, labels = ax_weight.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    ax_weight.legend(handles + handles2, labels + labels2, fontsize=8,
                     loc="upper left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_candidates(observed: np.ndarray, candidates: List[Candidate],
                    path: Path):
    """每个候选一行：观测 | 重建 | 纯物理检查 | 光阑。"""
    columns = ["Observed", "Reconstructed", "Physical check", "Aperture"]
    rows = max(len(candidates), 1)
    fig, axes = plt.subplots(rows, len(columns),
                             figsize=(3.0 * len(columns), 2.9 * rows),
                             squeeze=False)
    for row, candidate in enumerate(candidates):
        panels = [
            (observed, "inferno"),
            (candidate.pattern_pred, "inferno"),
            (candidate.pattern_physical, "inferno"),
            (candidate.aperture_soft, "gray"),
        ]
        for col, (image, cmap) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
            if row == 0:
                ax.set_title(columns[col], fontsize=10)
            ax.axis("off")
        axes[row, 0].set_ylabel(f"rank {candidate.rank}", fontsize=9)
        axes[row, 0].set_xlabel(
            f"MAE={candidate.pattern_mae:.2e}  "
            f"area={candidate.area_fraction:.2f}  "
            f"z={candidate.z_m:.2f}", fontsize=7)
    fig.suptitle("Top candidates: data-consistent aperture reconstructions",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_residual(candidate: Candidate, observed: np.ndarray, path: Path):
    residual = candidate.pattern_pred - observed
    limit = max(float(np.abs(residual).max()), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].imshow(observed, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title("Observed")
    axes[1].imshow(candidate.pattern_pred, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Reconstructed")
    image = axes[2].imshow(residual, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[2].set_title("Residual")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"rank {candidate.rank}  pattern MAE={candidate.pattern_mae:.3e}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_expected_vs_top1(observed: np.ndarray, reference_mask: np.ndarray,
                          reference_pattern: np.ndarray,
                          candidate: Candidate, reference_info: Dict,
                          path: Path):
    """预期效果 vs 实际 top1；reference 只用于画图和诊断。"""
    top_mask = candidate.aperture_bin
    top_display = candidate.aperture_soft
    top_pattern = candidate.pattern_pred
    overlay = np.zeros(reference_mask.shape + (3,), dtype=np.float32)
    overlay[..., 0] = reference_mask
    overlay[..., 1] = top_mask
    difference = np.abs(reference_mask - top_mask)
    residual = top_pattern - observed
    limit = max(float(np.abs(residual).max()), 1e-6)

    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.6))
    panels = [
        (axes[0, 0], reference_mask, "gray", "Expected aperture (reference)"),
        (axes[0, 1], top_display, "gray", "Top1 aperture (reconstruction)"),
        (axes[0, 2], overlay, None,
         "Overlay: red=expected, green=top1, yellow=both"),
        (axes[0, 3], difference, "magma", "|expected - top1|"),
        (axes[1, 0], reference_pattern, "inferno",
         "Expected pattern (reference forward)"),
        (axes[1, 1], observed, "inferno", "Observed pattern (input)"),
        (axes[1, 2], top_pattern, "inferno", "Top1 reconstructed pattern"),
        (axes[1, 3], residual, "RdBu_r", "Residual (top1 - observed)"),
    ]
    for ax, image, cmap, label in panels:
        if cmap is None:
            ax.imshow(np.clip(image, 0.0, 1.0))
        elif cmap == "RdBu_r":
            ax.imshow(image, cmap=cmap, vmin=-limit, vmax=limit)
        else:
            ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    note = (f"IoU={reference_info['iou_top1']:.3f}   "
            f"reference pattern MAE={reference_info['pattern_mae']:.2e}   "
            f"top1 MAE={candidate.pattern_mae:.2e}   "
            f"area reference={reference_info['area_fraction']:.3f} / "
            f"top1={candidate.area_fraction:.3f}")
    fig.suptitle("Expected vs top1 — reference used for comparison only, "
                 "never in loss / gradients / optimisation\n" + note,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_candidate(folder: Path, candidate: Candidate, observed: np.ndarray):
    prefix = f"rank{candidate.rank}"
    np.save(folder / f"{prefix}_aperture.npy", candidate.aperture_soft)
    np.save(folder / f"{prefix}_aperture_bin.npy", candidate.aperture_bin)
    np.save(folder / f"{prefix}_pattern_pred.npy", candidate.pattern_pred)
    np.save(folder / f"{prefix}_pattern_physical.npy", candidate.pattern_physical)
    save_image(folder / f"{prefix}_aperture.png", candidate.aperture_soft,
               "gray", 0.0, 1.0)
    save_image(folder / f"{prefix}_pattern_pred.png", candidate.pattern_pred,
               "inferno", 0.0, 1.0)
    save_image(folder / f"{prefix}_pattern_physical.png",
               candidate.pattern_physical, "inferno", 0.0, 1.0)
    write_history(folder / f"{prefix}_history.csv", candidate.history)
    plot_loss(candidate.history, folder / f"{prefix}_loss_curve.png",
              f"rank {candidate.rank}  data={candidate.data_loss:.2e}")
    plot_residual(candidate, observed, folder / f"{prefix}_residual.png")


# ============================================================
# 4.2 单张图反演
# ============================================================
def run_one(observed: torch.Tensor, forward: FrozenForward,
            cfg: InversionConfig, out: Path,
            sample_info: Optional[Dict] = None,
            reference: Optional[Dict] = None) -> Dict:
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    top, all_candidates = invert(observed, forward, cfg)
    observed_np = observed[0, 0].detach().cpu().numpy()

    # 参考光阑只在反演完成后用于作图/诊断，绝不进入优化。
    reference_info: Optional[Dict] = None
    reference_mask_np: Optional[np.ndarray] = None
    reference_pattern_np: Optional[np.ndarray] = None
    if reference is not None and top:
        with torch.no_grad():
            mask = reference["mask"].to(forward.device)
            support_ref, z_ref_self, cond_ref_self = forward.geometry_from_mask(
                mask, nf=cfg.nf, support_fraction=cfg.support_fraction)
            if cfg.geometry == "self-consistent":
                z_ref, condition_ref = z_ref_self, cond_ref_self
            else:
                # 与 top1 的最终评估保持同一套 z 规则。
                z_ref, condition_ref = _fixed_geometry(forward, cfg,
                                                       forward.device)
            pattern_ref = forward(mask, z_ref, condition_ref)[0, 0]
            reference_mask_np = mask[0, 0].cpu().numpy()
            reference_pattern_np = pattern_ref.cpu().numpy()
            pattern_mae_ref = float((pattern_ref - observed).abs().mean())
            iou_ref = centroid_iou(reference_mask_np, top[0].aperture_bin)
            area_ref = float(mask.mean())
            z_ref_value = float(z_ref.reshape(-1)[0])
        reference_info = {
            "source": reference.get("source"),
            "note": "仅用于对比与诊断，不参与 loss、梯度或任何优化步骤",
            "pattern_mae": pattern_mae_ref,
            "iou_top1": iou_ref,
            "area_fraction": area_ref,
            "z_m": z_ref_value,
            "figure": "expected_vs_top1.png",
        }

    warnings: List[str] = []
    for candidate in top:
        if candidate.condition_zscore > 3.0:
            warnings.append(
                f"rank{candidate.rank}: condition z-score="
                f"{candidate.condition_zscore:.2f} > 3，残差条件可能是外推")
        if candidate.residual_ratio > 0.5:
            warnings.append(
                f"rank{candidate.rank}: residual_ratio="
                f"{candidate.residual_ratio:.2f} 偏大，建议用 "
                "--residual-weight 0 对照")
        if candidate.physical_mae > 2.0 * candidate.pattern_mae + 1e-5:
            warnings.append(
                f"rank{candidate.rank}: physical MAE 明显大于 checkpoint MAE，"
                "可能存在残差利用或纯物理不一致")
        if candidate.area_fraction < 0.002 or candidate.area_fraction > 0.65:
            warnings.append(
                f"rank{candidate.rank}: area_fraction="
                f"{candidate.area_fraction:.3f} 超出常见范围")

    pairwise = {
        str(candidate.rank): {
            str(other.rank): round(centroid_iou(candidate.aperture_bin,
                                                other.aperture_bin), 4)
            for other in top if other is not candidate
        }
        for candidate in top
    }
    ambiguous = any(
        value < 0.5 for row in pairwise.values() for value in row.values())

    result = {
        "sample": sample_info or {},
        "checkpoint": forward.checkpoint_path,
        "nf": cfg.nf,
        "grid": cfg.grid,
        "mode": forward.mode,
        "geometry": cfg.geometry,
        "residual_weight": forward.residual_weight,
        "reference": reference_info,
        "config": {
            "scales": list(cfg.scales),
            "steps": list(cfg.steps),
            "lrs": list(cfg.lrs),
            "restarts": cfg.restarts,
            "top_k": cfg.top_k,
            "loss_schedule": cfg.loss_schedule,
            "fit_fraction": cfg.fit_fraction,
            "prior_fraction": cfg.prior_fraction,
            "detail_fraction": cfg.detail_fraction,
            "prior_ratio": cfg.prior_ratio,
            "lambda_coarse": cfg.lambda_coarse,
            "coarse_kernel": cfg.coarse_kernel,
            "sigma_coarse_scale": cfg.sigma_coarse_scale,
            "sigma_detail_scale": cfg.sigma_detail_scale,
            "edge_sigma": cfg.edge_sigma,
            "geometric_stage": cfg.geometric_stage,
            "geom_steps": cfg.geom_steps,
            "geom_primitive": cfg.geom_primitive,
            "geom_sides": cfg.geom_sides,
            "geom_lr": cfg.geom_lr,
            "geom_max_positive": cfg.geom_max_positive,
            "geom_max_negative": cfg.geom_max_negative,
            "temperature_end": cfg.temperature_end,
            "lambda_freq": cfg.lambda_freq,
            "lambda_corr": cfg.lambda_corr,
            "lambda_tv": cfg.lambda_tv,
            "lambda_lap": cfg.lambda_lap,
            "lambda_perim": cfg.lambda_perim,
            "lambda_bin": cfg.lambda_bin,
            "lambda_open": cfg.lambda_open,
            "lambda_open5": cfg.lambda_open5,
            "lambda_island": cfg.lambda_island,
            "lambda_area": cfg.lambda_area,
            "lambda_area_center": cfg.lambda_area_center,
            "lambda_res": cfg.lambda_res,
            "smooth_sigma": cfg.smooth_sigma,
            "island_kernel": cfg.island_kernel,
            "island_min_mass": cfg.island_min_mass,
            "primitive_fit": cfg.primitive_fit,
            "primitive_min_area": cfg.primitive_min_area,
            "primitive_max_fit_area": cfg.primitive_max_fit_area,
            "primitive_min_solidity": cfg.primitive_min_solidity,
            "support_fraction": cfg.support_fraction,
        },
        "candidates": [candidate.summary() for candidate in top],
        "all_candidates": [candidate.summary() for candidate in all_candidates],
        "pairwise_iou": pairwise,
        "ambiguous": ambiguous,
        "warnings": warnings,
        "runtime_s": time.perf_counter() - started,
    }
    write_json(out / "result.json", result)
    write_json(out / "config.json", {
        "sample": sample_info or {},
        "checkpoint": forward.checkpoint_path,
        "nf": cfg.nf,
        "mode": forward.mode,
        "geometry": cfg.geometry,
        "inversion_config": asdict(cfg),
    })
    plot_candidates(observed_np, top, out / "comparison.png")
    if reference_info is not None:
        plot_expected_vs_top1(observed_np, reference_mask_np,
                              reference_pattern_np, top[0],
                              reference_info, out / "expected_vs_top1.png")
        np.save(out / "expected_aperture_bin.npy", reference_mask_np)
        np.save(out / "expected_pattern_pred.npy", reference_pattern_np)
        write_json(out / "reference.json", reference_info)
    for candidate in top:
        save_candidate(out, candidate, observed_np)
    return result


# ============================================================
# 4.3 CLI
# ============================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pattern", default=None,
                        help="单张衍射图样 .npy/.png（网络唯一输入）")
    parser.add_argument("--write-cases", default=None, metavar="DIR",
                        help="只生成测试用例（pattern/gt/meta）后退出，不做反演")
    parser.add_argument("--case-shapes", nargs="+", default=list(CASE_BUILDERS),
                        choices=list(CASE_BUILDERS),
                        help="--write-cases 时要生成的用例")
    parser.add_argument("--case-scale", type=float, default=0.5,
                        help="--write-cases 时光阑相对画幅的缩放")
    parser.add_argument("--case-grid", type=int, default=256,
                        help="--write-cases 时的网格大小")
    parser.add_argument("--checkpoint", help="直接指定 physics best_model.pt")
    parser.add_argument("--run", help="train.py 输出目录，如 runs/v1")
    parser.add_argument("--nf", type=float, default=None,
                        help="已知菲涅耳数；单图 + 直接 checkpoint 时可省略")
    parser.add_argument("--out", default=None,
                        help="输出目录；默认 inverse_runs/<时间戳>，必须不存在")

    parser.add_argument("--forward-mode", choices=("checkpoint", "physical"),
                        default="checkpoint",
                        help="checkpoint: FFT+残差；physical: 纯 FFT/ASM 对照")
    parser.add_argument("--residual-weight", type=float, default=1.0,
                        help="checkpoint 模式下残差项权重，0 表示只用传播层")
    parser.add_argument("--geometry",
                        choices=("self-consistent", "canonical", "known"),
                        default=None,
                        help="z 的来源；默认按图样自洽。光阑仿真模式默认固定合成时的 z")
    parser.add_argument("--z", type=float, default=None,
                        help="已知传播距离（米）；配合 --geometry known")
    parser.add_argument("--support-fraction", type=float, default=0.5,
                        help="canonical 模式的参考支撑占画幅比例")
    parser.add_argument("--input-mode", choices=("dataset", "image"),
                        default="dataset",
                        help="dataset: 已是 [0,1] log 图样；image: 原始强度图")
    parser.add_argument("--invert-image", action="store_true",
                        help="PNG 输入反相")

    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--scales", type=int, nargs="+", default=(64, 128, 256))
    parser.add_argument("--steps", type=int, nargs="+", default=(240, 240, 480))
    parser.add_argument("--lrs", type=float, nargs="+", default=(0.05, 0.02, 0.01))
    parser.add_argument("--loss-schedule", choices=("phased", "fixed"),
                        default="phased",
                        help="phased: 前期只拟合数据、后期收形状；fixed: 旧固定权重")
    parser.add_argument("--fit-fraction", type=float, default=0.30,
                        help="该进度之前只用数据项拟合大方向")
    parser.add_argument("--prior-fraction", type=float, default=0.60,
                        help="该进度时平滑/面积/连通性先验到满权重")
    parser.add_argument("--detail-fraction", type=float, default=0.85,
                        help="该进度时二值/形态学先验到满权重")
    parser.add_argument("--prior-ratio", type=float, default=0.25,
                        help="先验总权重相对空间 L1 的目标占比")
    parser.add_argument("--lambda-coarse", type=float, default=0.25,
                        help="前期低频大方向数据项权重")
    parser.add_argument("--coarse-kernel", type=int, default=9,
                        help="低频数据项的平均池化核，>=3 且为奇数")
    parser.add_argument("--sigma-coarse-scale", type=float, default=1.6,
                        help="粗阶段平滑尺度相对 --smooth-sigma 的倍数")
    parser.add_argument("--sigma-detail-scale", type=float, default=1.0,
                        help="细节阶段平滑尺度相对 --smooth-sigma 的倍数")
    parser.add_argument("--edge-sigma", type=float, default=1.5,
                        help="结束后对光阑做边缘平滑的 sigma（像素），0=关闭")
    parser.add_argument("--geometric-stage", choices=("on", "off"),
                        default="on",
                        help="后段是否把像素解拟合成平滑椭圆基元")
    parser.add_argument("--geom-primitive", choices=("ellipse", "polygon"),
                        default="ellipse",
                        help="几何基元形状；polygon 为正多边形")
    parser.add_argument("--geom-sides", type=int, default=6,
                        help="polygon 模式的边数（>=3）")
    parser.add_argument("--geom-steps", type=int, default=300,
                        help="几何参数化阶段的迭代步数")
    parser.add_argument("--geom-lr", type=float, default=0.02,
                        help="几何基元参数的 Adam 学习率")
    parser.add_argument("--geom-max-pos", type=int, default=8,
                        help="最多正椭圆基元数")
    parser.add_argument("--geom-max-neg", type=int, default=4,
                        help="最多孔洞（负椭圆）数")
    parser.add_argument("--geom-softness", type=float, default=25.0,
                        help="椭圆边界 sigmoid 软硬，越大越接近二值")
    parser.add_argument("--expected", default=None,
                        help="预期光阑 .npy/.png；只在结束后作图对比，不参与优化")
    parser.add_argument("--expected-sample", type=int, default=None,
                        help="从 --dataset 的 apertures.npy 取第 N 个样本作为预期光阑")
    parser.add_argument("--dataset", default=None,
                        help="数据目录（配合 --expected-sample），如 data/v1")
    parser.add_argument("--lambda-freq", type=float, default=0.02)
    parser.add_argument("--lambda-corr", type=float, default=0.5,
                        help="相关系数项权重；尺度不变，抑制空光阑/大光斑平凡解")
    parser.add_argument("--lambda-tv", type=float, default=1e-3)
    parser.add_argument("--lambda-lap", type=float, default=3e-3,
                        help="边界曲率/锯齿惩罚")
    parser.add_argument("--lambda-perim", type=float, default=1e-3,
                        help="周长/面积比惩罚")
    parser.add_argument("--lambda-bin", type=float, default=1e-3)
    parser.add_argument("--lambda-open", type=float, default=1e-3)
    parser.add_argument("--lambda-open5", type=float, default=0.0,
                        help="5x5 开运算，进一步惩罚小于 5 像素的细碎结构")
    parser.add_argument("--lambda-island", type=float, default=3e-2,
                        help="孤立小岛惩罚，抑制散点/毛刺")
    parser.add_argument("--island-kernel", type=int, default=7)
    parser.add_argument("--island-min-mass", type=float, default=6.0)
    parser.add_argument("--lambda-area", type=float, default=1e-2)
    parser.add_argument("--lambda-area-center", type=float, default=0.1,
                        help="弱对数面积目标；把面积拉向该 restart 的初始面积")
    parser.add_argument("--lambda-res", type=float, default=1e-3)
    parser.add_argument("--smooth-sigma", type=float, default=1.0,
                        help="水平集高斯平滑，限制边界曲率/最小特征尺寸")
    parser.add_argument("--primitive-fit", choices=("none", "ellipse", "rect"),
                        default="none",
                        help="把连通域拟合成椭圆/矩形；默认关闭，避免抹掉孔洞/细结构")
    parser.add_argument("--primitive-min-area", type=int, default=20)
    parser.add_argument("--primitive-max-fit-area", type=int, default=4000)
    parser.add_argument("--primitive-min-solidity", type=float, default=0.2)
    parser.add_argument("--primitive-max-components", type=int, default=32)
    parser.add_argument("--primitive-shrink", type=float, default=1.0)
    parser.add_argument("--geometry-every", type=int, default=10)
    parser.add_argument("--temperature-end", type=float, default=0.35,
                        help="结束时 sigmoid 温度；越大边缘越软")
    parser.add_argument("--ste-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.write_cases:
        nf = float(args.nf if args.nf is not None else 1.0)
        write_cases(args.write_cases, args.case_shapes, nf,
                    args.case_grid, args.case_scale)
        return
    if not args.pattern:
        raise SystemExit("必须提供 --pattern（或用 --write-cases 只生成用例）")
    if args.restarts < 1 or args.top_k < 1:
        raise SystemExit("--restarts 与 --top-k 必须为正")
    if args.top_k > args.restarts:
        args.top_k = args.restarts
    if args.forward_mode == "physical" and args.residual_weight != 1.0:
        raise SystemExit("--forward-mode physical 时 --residual-weight 无效")
    if not args.checkpoint and not args.run:
        raise SystemExit("必须提供 --checkpoint 或 --run")
    if args.run and args.nf is None:
        raise SystemExit("用 --run 查找 checkpoint 时必须提供 --nf")
    if args.geometry == "known" and args.z is None:
        raise SystemExit("--geometry known 需要 --z（米）")
    if args.expected and args.expected_sample is not None:
        raise SystemExit("--expected 与 --expected-sample 只能选一个")
    if args.expected_sample is not None and not args.dataset:
        raise SystemExit("--expected-sample 需要 --dataset（如 data/v1）")

    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    out = (Path(args.out).resolve() if args.out
           else (Path("inverse_runs") / time.strftime("%Y%m%d_%H%M%S")).resolve())
    if out.exists():
        raise SystemExit(f"输出目录已存在：{out}；请换一个新名字")
    out.mkdir(parents=True)

    checkpoint = resolve_checkpoint(
        run=args.run, nf=args.nf, model="physics", checkpoint=args.checkpoint)
    forward = FrozenForward(
        checkpoint, mode=args.forward_mode, device=device,
        residual_weight=args.residual_weight, nf=args.nf)
    if args.nf is not None and abs(float(args.nf) - float(forward.nf)) > 1e-9:
        raise SystemExit(
            f"--nf={args.nf} 与 checkpoint 的 NF={forward.nf} 不一致")
    nf = float(forward.nf if args.nf is None else args.nf)
    grid = forward.config.grid
    geometry = args.geometry or "self-consistent"

    cfg = InversionConfig(
        nf=nf,
        grid=grid,
        scales=tuple(args.scales),
        steps=tuple(args.steps),
        lrs=tuple(args.lrs),
        restarts=args.restarts,
        top_k=args.top_k,
        geometry=geometry,
        support_fraction=args.support_fraction,
        z_m=args.z,
        geometry_every=args.geometry_every,
        ste_steps=args.ste_steps,
        temperature_end=args.temperature_end,
        loss_schedule=args.loss_schedule,
        fit_fraction=args.fit_fraction,
        prior_fraction=args.prior_fraction,
        detail_fraction=args.detail_fraction,
        prior_ratio=args.prior_ratio,
        lambda_coarse=args.lambda_coarse,
        coarse_kernel=args.coarse_kernel,
        sigma_coarse_scale=args.sigma_coarse_scale,
        sigma_detail_scale=args.sigma_detail_scale,
        edge_sigma=args.edge_sigma,
        geometric_stage=(args.geometric_stage == "on"),
        geom_max_positive=args.geom_max_pos,
        geom_max_negative=args.geom_max_neg,
        geom_steps=args.geom_steps,
        geom_primitive=args.geom_primitive,
        geom_sides=args.geom_sides,
        geom_lr=args.geom_lr,
        geom_softness=args.geom_softness,
        lambda_freq=args.lambda_freq,
        lambda_corr=args.lambda_corr,
        lambda_tv=args.lambda_tv,
        lambda_lap=args.lambda_lap,
        lambda_perim=args.lambda_perim,
        lambda_bin=args.lambda_bin,
        lambda_open=args.lambda_open,
        lambda_open5=args.lambda_open5,
        lambda_island=args.lambda_island,
        lambda_area=args.lambda_area,
        lambda_area_center=args.lambda_area_center,
        lambda_res=args.lambda_res,
        smooth_sigma=args.smooth_sigma,
        island_kernel=args.island_kernel,
        island_min_mass=args.island_min_mass,
        primitive_fit=args.primitive_fit,
        primitive_min_area=args.primitive_min_area,
        primitive_max_fit_area=args.primitive_max_fit_area,
        primitive_min_solidity=args.primitive_min_solidity,
        primitive_max_components=args.primitive_max_components,
        primitive_shrink=args.primitive_shrink,
        seed=args.seed,
        verbose=True,
    )

    print(f"设备：{device}")
    print(f"checkpoint：{checkpoint}")
    print(f"NF={nf:g}  grid={grid}  mode={args.forward_mode}  "
          f"geometry={geometry}")

    observed = load_pattern(args.pattern, grid=grid,
                            input_mode=args.input_mode,
                            invert=args.invert_image).to(device)
    sample_info = {"pattern": str(Path(args.pattern).resolve()), "nf": nf}
    reference = None
    if args.expected_sample is not None:
        reference = {
            "mask": load_reference_sample(args.dataset,
                                          args.expected_sample, grid),
            "source": {"dataset": str(Path(args.dataset).resolve()),
                       "index": int(args.expected_sample)},
        }
    elif args.expected:
        reference = {
            "mask": load_reference_aperture(args.expected, grid),
            "source": str(Path(args.expected).resolve()),
        }
    if reference is not None:
        print("配对光阑只用于 expected_vs_top1.png 与诊断，"
              "不进入 loss、梯度或任何网络输入")

    run_one(observed, forward, cfg, out, sample_info=sample_info,
            reference=reference)
    print(f"完成：{out}")


if __name__ == "__main__":
    main()
