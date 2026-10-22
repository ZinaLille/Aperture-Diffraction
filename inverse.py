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
from PIL import Image
from torch import nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate import PhysicsConfig, group_name, postprocess
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
    steps: Tuple[int, ...] = (120, 120, 200)
    lrs: Tuple[float, ...] = (0.05, 0.02, 0.01)
    restarts: int = 4
    top_k: int = 3
    geometry: str = "self-consistent"   # self-consistent | canonical | known
    support_fraction: float = 0.5
    z_m: Optional[float] = None
    geometry_every: int = 10

    temperature_start: float = 1.0
    temperature_end: float = 0.1
    ste_steps: int = 80

    lambda_freq: float = 0.1
    lambda_tv: float = 1e-3
    lambda_lap: float = 1e-3
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
    primitive_fit: str = "ellipse"       # none | ellipse | rect
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
               self.lambda_area_center, self.lambda_res) < 0:
            raise ValueError("损失权重不能为负")
        if self.island_kernel < 3 or self.island_kernel % 2 == 0:
            raise ValueError("island_kernel 必须是 >=3 的奇数")
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
        }


# ============================================================
# 3.2 可微物理先验
# ============================================================
def pattern_loss(pred: torch.Tensor, target: torch.Tensor,
                 lambda_freq: float = 0.1):
    """L1 + log 频谱 L1，和 train.py 的标签域一致。"""
    l1 = F.l1_loss(pred, target)
    if lambda_freq > 0:
        f_pred = torch.fft.fft2(pred[:, 0])
        f_target = torch.fft.fft2(target[:, 0])
        freq = F.l1_loss(torch.log1p(f_pred.abs()),
                         torch.log1p(f_target.abs()))
    else:
        freq = pred.new_zeros(())
    return l1 + lambda_freq * freq, l1, freq


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


def prior_schedule(progress: float) -> Tuple[float, float]:
    """二值项全程渐强；开运算项后 85% 才逐渐加入。"""
    progress = float(np.clip(progress, 0.0, 1.0))
    binary_weight = 0.2 + 0.8 * progress
    opening_weight = max(0.0, (progress - 0.15) / 0.85)
    return binary_weight, opening_weight


def prior_loss(aperture: torch.Tensor, cfg: InversionConfig,
               progress: float, area_target: Optional[float] = None):
    tv = tv_loss(aperture)
    lap = laplacian_loss(aperture)
    perimeter = perimeter_ratio(aperture)
    binary = binary_loss(aperture)
    opening = opening_loss(aperture, kernel=3)
    opening5 = opening_loss(aperture, kernel=5)
    island = island_loss(aperture, kernel=cfg.island_kernel,
                         min_mass=cfg.island_min_mass)
    area = area_hinge(aperture, cfg.area_low, cfg.area_high)
    binary_weight, opening_weight = prior_schedule(progress)
    total = (cfg.lambda_tv * tv +
             cfg.lambda_lap * lap +
             cfg.lambda_perim * perimeter +
             cfg.lambda_bin * binary_weight * binary +
             cfg.lambda_open * opening_weight * opening +
             cfg.lambda_open5 * opening_weight * opening5 +
             cfg.lambda_island * island +
             cfg.lambda_area * area)
    if cfg.lambda_area_center > 0 and area_target is not None:
        area_value = aperture.mean(dim=(1, 2, 3)).clamp_min(1e-6)
        area_center = (torch.log(area_value) -
                       math.log(max(float(area_target), 1e-6))).square().mean()
        total = total + cfg.lambda_area_center * area_center
    else:
        area_center = aperture.new_zeros(())
    stats = {
        "tv": float(tv.detach()),
        "lap": float(lap.detach()),
        "perim": float(perimeter.detach()),
        "bin": float(binary.detach()),
        "open": float(opening.detach()),
        "open5": float(opening5.detach()),
        "island": float(island.detach()),
        "area": float(area.detach()),
        "area_center": float(area_center.detach()),
        "bin_w": binary_weight,
        "open_w": opening_weight,
    }
    return total, stats


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
        self.smooth_sigma = float(smooth_sigma)
        self.blur_radius = max(1, int(math.ceil(3.0 * self.smooth_sigma))) \
            if self.smooth_sigma > 0 else 0
        if self.blur_radius > 0:
            coords = torch.arange(-self.blur_radius, self.blur_radius + 1,
                                  dtype=torch.float32)
            kernel = torch.exp(-coords.square() /
                               (2.0 * self.smooth_sigma ** 2))
            kernel = kernel / kernel.sum()
            kernel = (kernel[:, None] * kernel[None, :])[None, None]
            kernel = kernel.repeat(theta.shape[1], 1, 1, 1)
            self.register_buffer("blur_kernel", kernel, persistent=False)
        else:
            self.register_buffer("blur_kernel", torch.empty(0), persistent=False)

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

    for stage, (resolution, steps, lr) in enumerate(
            zip(cfg.scales, cfg.steps, cfg.lrs)):
        if stage > 0:
            # 不要原地改 Parameter.data 的形状，否则 autograd 仍记着旧形状。
            field = PixelField(
                resize_theta(field.theta.detach(), resolution),
                smooth_sigma=cfg.smooth_sigma).to(device)
        optimizer = torch.optim.Adam([field.theta], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(steps, 1))
        final_stage = stage == len(cfg.scales) - 1
        ste_start = max(0, steps - min(cfg.ste_steps, steps))

        for step in range(steps):
            progress = (global_step + step) / max(total_steps - 1, 1)
            field.temperature = (cfg.temperature_start +
                                 (cfg.temperature_end - cfg.temperature_start)
                                 * progress)
            aperture = field(cfg.grid)

            if cfg.geometry == "self-consistent":
                if z is None or step % cfg.geometry_every == 0:
                    _, z, condition = _update_geometry(
                        forward, cfg, aperture, z)
            elif z is None:
                z, condition = _fixed_geometry(forward, cfg, device)

            use_ste = final_stage and step >= ste_start
            if use_ste:
                hard = (aperture > 0.5).float()
                aperture_used = aperture + (hard - aperture).detach()
            else:
                aperture_used = aperture

            pred, simulated, delta = forward(
                aperture_used, z, condition, return_aux=True)
            data, l1, freq = pattern_loss(pred, observed, cfg.lambda_freq)
            prior, prior_stats = prior_loss(aperture, cfg, progress,
                                            area_target=area_fraction)
            residual = delta.abs().mean()
            loss = data + prior + cfg.lambda_res * residual

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
                    "prior": float(prior.detach()),
                    "tv": prior_stats["tv"],
                    "lap": prior_stats["lap"],
                    "perim": prior_stats["perim"],
                    "bin": prior_stats["bin"],
                    "open": prior_stats["open"],
                    "open5": prior_stats["open5"],
                    "island": prior_stats["island"],
                    "area": prior_stats["area"],
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
                aperture_bin = torch.from_numpy(cleaned)[None, None].to(device)
                aperture_soft = aperture_bin
                primitive_components = int(
                    primitive_info.get("components_fit", 0) +
                    primitive_info.get("components_kept", 0))
            else:
                primitive_info["fallback"] = "kept_raw_pixel_mask"
                primitive_components = int(
                    primitive_info.get("components_in", 0))
        support, z_final, condition_final = forward.geometry_from_mask(
            aperture_bin, nf=cfg.nf, support_fraction=cfg.support_fraction)
        pred_final, simulated_final, delta_final = forward(
            aperture_bin, z_final, condition_final, return_aux=True)
        physical_final = forward.physical_pattern(aperture_bin, z_final)
        final_progress = 1.0
        final_prior, _ = prior_loss(aperture_soft, cfg, final_progress,
                                    area_target=float(aperture_bin.mean()))
        final_data, _, _ = pattern_loss(pred_final, observed, cfg.lambda_freq)
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
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in (("loss", "total"), ("data", "data"),
                       ("prior", "prior")):
        values = [max(float(row.get(key, 0.0)), 1e-12) for row in history]
        ax.semilogy(steps, values, label=label, lw=1.4)
    ax.set(xlabel="step", ylabel="loss (floor 1e-12)", title=title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
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
            (candidate.aperture_bin, "gray"),
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


def save_candidate(folder: Path, candidate: Candidate, observed: np.ndarray):
    prefix = f"rank{candidate.rank}"
    np.save(folder / f"{prefix}_aperture.npy", candidate.aperture_soft)
    np.save(folder / f"{prefix}_aperture_bin.npy", candidate.aperture_bin)
    np.save(folder / f"{prefix}_pattern_pred.npy", candidate.pattern_pred)
    np.save(folder / f"{prefix}_pattern_physical.npy", candidate.pattern_physical)
    save_image(folder / f"{prefix}_aperture.png", candidate.aperture_bin,
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
            sample_info: Optional[Dict] = None) -> Dict:
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    top, all_candidates = invert(observed, forward, cfg)
    observed_np = observed[0, 0].detach().cpu().numpy()

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
        "config": {
            "scales": list(cfg.scales),
            "steps": list(cfg.steps),
            "lrs": list(cfg.lrs),
            "restarts": cfg.restarts,
            "top_k": cfg.top_k,
            "lambda_freq": cfg.lambda_freq,
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
    parser.add_argument("--pattern", required=True,
                        help="单张图样 .npy/.png")
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
                        default="self-consistent",
                        help="z 的来源；默认由当前光阑支撑和 NF 自洽计算")
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
    parser.add_argument("--steps", type=int, nargs="+", default=(120, 120, 200))
    parser.add_argument("--lrs", type=float, nargs="+", default=(0.05, 0.02, 0.01))
    parser.add_argument("--lambda-freq", type=float, default=0.1)
    parser.add_argument("--lambda-tv", type=float, default=1e-3)
    parser.add_argument("--lambda-lap", type=float, default=1e-3,
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
                        default="ellipse",
                        help="把连通域拟合成椭圆/矩形，去掉小岛并平滑边界")
    parser.add_argument("--primitive-min-area", type=int, default=20)
    parser.add_argument("--primitive-max-fit-area", type=int, default=4000)
    parser.add_argument("--primitive-min-solidity", type=float, default=0.2)
    parser.add_argument("--primitive-max-components", type=int, default=32)
    parser.add_argument("--primitive-shrink", type=float, default=1.0)
    parser.add_argument("--geometry-every", type=int, default=10)
    parser.add_argument("--ste-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.restarts < 1 or args.top_k < 1:
        raise SystemExit("--restarts 与 --top-k 必须为正")
    if args.top_k > args.restarts:
        args.top_k = args.restarts
    if args.forward_mode == "physical" and args.residual_weight != 1.0:
        raise SystemExit("--forward-mode physical 时 --residual-weight 无效")
    if args.geometry == "known" and args.z is None:
        raise SystemExit("--geometry known 需要 --z（米）")
    if not args.checkpoint and not args.run:
        raise SystemExit("必须提供 --checkpoint 或 --run")
    if args.run and args.nf is None:
        raise SystemExit("用 --run 查找 checkpoint 时必须提供 --nf")

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

    cfg = InversionConfig(
        nf=nf,
        grid=grid,
        scales=tuple(args.scales),
        steps=tuple(args.steps),
        lrs=tuple(args.lrs),
        restarts=args.restarts,
        top_k=args.top_k,
        geometry=args.geometry,
        support_fraction=args.support_fraction,
        z_m=args.z,
        geometry_every=args.geometry_every,
        ste_steps=args.ste_steps,
        lambda_freq=args.lambda_freq,
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
          f"geometry={args.geometry}")

    observed = load_pattern(args.pattern, grid=grid,
                            input_mode=args.input_mode,
                            invert=args.invert_image).to(device)
    run_one(observed, forward, cfg, out,
            sample_info={"pattern": str(Path(args.pattern).resolve()),
                         "nf": nf})
    print(f"完成：{out}")


if __name__ == "__main__":
    main()
