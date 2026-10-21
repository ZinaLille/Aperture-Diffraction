"""阶段一训练：每个菲涅耳数分组训练 physics；baseline 为可选对照。

physics  = 固定传播层 + 零初始化残差 U-Net
baseline = 纯 U-Net（相同骨干，无物理层），需要时用 --models baseline 启用
损失     = L1 + λ_f·log 频谱 L1 + λ_s·(1-SSIM) + λ_r·|ΔI|
SSIM 为内置可微实现，不依赖 torchmetrics / pytorch-msssim。

用法：
    python train.py --data data/v1 --out runs/v1
    python train.py --data data/v1 --models physics baseline --out runs/with_baseline
    python train.py --data data/v1 --groups 3 5 --propagator asm \
        --epochs 10 --batch-size 4 --out runs/asm_v1

每个模型目录输出：
    best_model.pt  history.csv  selection.json  test_metrics.json
    loss_curve.png  train_result.png
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate import PhysicsConfig, image_metrics, support_width


# ============================================================
# 1. 网络与损失
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


def ssim(pred, target, window=11, sigma=1.5):
    """内置可微 SSIM；输入 (B,1,H,W) 已归一化到 [0,1]。

    使用高斯窗、非重叠填充，返回 batch 平均值。尺寸小于 11 时自动缩小窗。
    """
    height, width = pred.shape[-2:]
    window = min(window, height, width)
    if window % 2 == 0:
        window -= 1
    coords = torch.arange(window, dtype=pred.dtype, device=pred.device) - window // 2
    gaussian = torch.exp(-(coords**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :])[None, None]
    kernel = kernel.expand(pred.shape[1], 1, window, window).contiguous()

    mu_p = F.conv2d(pred, kernel, padding=window // 2, groups=pred.shape[1])
    mu_t = F.conv2d(target, kernel, padding=window // 2, groups=target.shape[1])
    var_p = (F.conv2d(pred * pred, kernel, padding=window // 2,
                      groups=pred.shape[1]) - mu_p**2).clamp_min(0)
    var_t = (F.conv2d(target * target, kernel, padding=window // 2,
                      groups=target.shape[1]) - mu_t**2).clamp_min(0)
    cov_pt = F.conv2d(pred * target, kernel, padding=window // 2,
                      groups=pred.shape[1]) - mu_p * mu_t
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_p * mu_t + c1) * (2 * cov_pt + c2) /
             ((mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2)))
    return value.mean()


class Loss(nn.Module):
    """L = L1 + λ_f·L1(log|FFT|) + λ_s·(1-SSIM) + λ_r·|ΔI|。"""

    def __init__(self, lambda_freq=0.1, lambda_ssim=0.0, lambda_res=1e-3,
                 use_log_freq=True):
        super().__init__()
        self.lambda_freq = lambda_freq
        self.lambda_ssim = lambda_ssim
        self.lambda_res = lambda_res
        self.use_log_freq = use_log_freq
        self.l1 = nn.L1Loss()

    def forward(self, pred, target, delta=None):
        loss_data = self.l1(pred, target)
        if self.lambda_freq > 0:
            a_pred = torch.abs(torch.fft.fft2(pred.squeeze(1)))
            a_target = torch.abs(torch.fft.fft2(target.squeeze(1)))
            if self.use_log_freq:
                a_pred, a_target = torch.log1p(a_pred), torch.log1p(a_target)
            loss_freq = torch.mean(torch.abs(a_pred - a_target))
        else:
            loss_freq = pred.new_zeros(())
        loss_ssim = (1.0 - ssim(pred, target)) if self.lambda_ssim > 0 else pred.new_zeros(())
        loss_res = (torch.mean(torch.abs(delta))
                    if self.lambda_res > 0 and delta is not None else pred.new_zeros(()))
        total = (loss_data + self.lambda_freq * loss_freq
                 + self.lambda_ssim * loss_ssim + self.lambda_res * loss_res)
        stats = {"l1": loss_data.item(), "freq": loss_freq.item(),
                 "ssim": loss_ssim.item(), "res": loss_res.item()}
        return total, stats


class ConditionedUNet(nn.Module):
    def __init__(self, in_channels, base_ch=32, residual=False, target_mean=0.0):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, 2)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 1)
        if residual:
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)
        else:
            nn.init.normal_(self.out_conv.weight, 0, 1e-3)
            nn.init.constant_(self.out_conv.bias, target_mean)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        x = self.bottleneck(self.pool(e3))
        x = self.dec3(torch.cat((self.up3(x), e3), 1))
        x = self.dec2(torch.cat((self.up2(x), e2), 1))
        x = self.dec1(torch.cat((self.up1(x), e1), 1))
        return self.out_conv(x)


class FraunhoferDetectorLayer(nn.Module):
    """固定夫琅禾费层：FFT → 模平方 → 峰值归一化 → log 压缩。"""

    def __init__(self, alpha=100.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, aperture, z_m=None):
        field = torch.fft.fft2(aperture.squeeze(1))
        intensity = torch.fft.fftshift(field, dim=(-2, -1)).abs().square()
        intensity = intensity / intensity.amax((-2, -1), keepdim=True).clamp_min(1e-30)
        intensity = torch.log1p(self.alpha * intensity)
        intensity = intensity / intensity.amax((-2, -1), keepdim=True).clamp_min(1e-30)
        return intensity.unsqueeze(1)


class AngularSpectrumDetectorLayer(nn.Module):
    """固定可微缩放角谱层；无参数，接收以米为单位的 z。"""

    def __init__(self, config, precision="float32"):
        super().__init__()
        self.config = config
        if precision not in ("float32", "float64"):
            raise ValueError("ASM 精度必须是 float32 或 float64")
        self.real_dtype = torch.float64 if precision == "float64" else torch.float32
        self.complex_dtype = torch.complex128 if precision == "float64" else torch.complex64
        self.alpha = config.log_alpha
        c, pad, n = config.oversample, config.pad_factor, config.grid
        self.fine_grid = n * c
        self.padded_grid = m = self.fine_grid * pad
        dx = config.input_pitch_m / c
        frequency = torch.fft.fftfreq(m, d=dx, dtype=torch.float64)
        q = config.wavelength_m**2 * (frequency[:, None]**2 + frequency[None, :]**2)
        root = torch.sqrt(torch.clamp(1 - q, min=0))
        coefficient = -(2 * torch.pi / config.wavelength_m) * q.square() / (2 * (1 + root).square())
        coord = (torch.arange(m, dtype=torch.float64) - m // 2 - (c - 1) / 2) * dx
        self.register_buffer("correction_coefficient", coefficient, persistent=False)
        self.register_buffer("propagating", q < 1, persistent=False)
        self.register_buffer("chirp_coefficient",
                             torch.pi * coord.square() / config.wavelength_m, persistent=False)
        self.register_buffer("output_indices",
                             m // 2 + (torch.arange(n) - n // 2) * pad, persistent=False)

    def raw_intensity(self, aperture, z_m):
        if aperture.ndim != 4 or aperture.shape[1:] != (1, self.config.grid, self.config.grid):
            raise ValueError("ASM 输入必须是 [B,1,grid,grid]")
        if z_m is None:
            raise ValueError("ASM 需要以米为单位的 z")
        z = torch.as_tensor(z_m, dtype=torch.float64, device=aperture.device).reshape(-1)
        if z.numel() != len(aperture) or not torch.isfinite(z).all() or (z <= 0).any():
            raise ValueError("每个样本需要一个有限正传播距离")
        c = self.config.oversample
        fine = aperture[:, 0].to(self.real_dtype).repeat_interleave(c, -2).repeat_interleave(c, -1)
        offset = (self.padded_grid - self.fine_grid) // 2
        fine = F.pad(fine, (offset, offset, offset, offset))
        spectrum = torch.fft.fft2(torch.fft.ifftshift(fine, dim=(-2, -1)))
        phase = z[:, None, None] * self.correction_coefficient
        correction = torch.polar(torch.ones_like(phase), phase).to(self.complex_dtype)
        corrected = torch.fft.fftshift(
            torch.fft.ifft2(spectrum * (correction * self.propagating)), dim=(-2, -1))
        phase = self.chirp_coefficient[None, :] / z[:, None]
        chirp = torch.polar(torch.ones_like(phase), phase).to(self.complex_dtype)
        field = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(
            corrected * chirp[:, :, None] * chirp[:, None, :], dim=(-2, -1))), dim=(-2, -1))
        field = field.index_select(-2, self.output_indices).index_select(-1, self.output_indices)
        scale = (self.config.input_pitch_m / c)**2 / (self.config.wavelength_m * z)
        return (field.abs().square() * scale[:, None, None].to(self.real_dtype).square()).unsqueeze(1)

    def forward(self, aperture, z_m):
        raw = self.raw_intensity(aperture, z_m)
        value = raw / raw.amax((-2, -1), keepdim=True).clamp_min(1e-30)
        value = torch.log1p(self.alpha * value)
        value = value / value.amax((-2, -1), keepdim=True).clamp_min(1e-30)
        return value.to(aperture.dtype)


class ForwardModel(nn.Module):
    def __init__(self, kind, base_ch=32, alpha=100.0, target_mean=0.0,
                 propagator="fraunhofer", physics_config=None, asm_precision="float32"):
        super().__init__()
        if kind not in ("physics", "baseline"):
            raise ValueError(kind)
        if propagator not in ("fraunhofer", "asm"):
            raise ValueError(propagator)
        self.kind = kind
        self.propagator = propagator
        config = physics_config or PhysicsConfig(log_alpha=alpha)
        self.physics = (AngularSpectrumDetectorLayer(config, asm_precision)
                        if propagator == "asm" and kind == "physics"
                        else FraunhoferDetectorLayer(alpha))
        self.network = ConditionedUNet(4 if kind == "physics" else 3, base_ch,
                                       residual=kind == "physics", target_mean=target_mean)

    def forward(self, aperture, condition, z_m=None):
        maps = condition[:, :, None, None].expand(-1, -1, *aperture.shape[-2:])
        if self.kind == "physics":
            simulated = self.physics(aperture, z_m)
            delta = self.network(torch.cat((aperture, simulated, maps), 1))
            return simulated + delta, simulated, delta
        predicted = self.network(torch.cat((aperture, maps), 1))
        return predicted, None, None


# ============================================================
# 2. 数据读取
# ============================================================
_GEOMETRY_CACHE = {}


def geometry_arrays(root, group):
    """按组缓存 z 与观察面像素间距；由光阑和 NF 确定性计算。"""
    key = (str(Path(root).resolve()), group)
    if key not in _GEOMETRY_CACHE:
        manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
        config = PhysicsConfig(**manifest["config"])
        nf = next(g["nf"] for g in manifest["groups"] if g["name"] == group)
        apertures = np.load(Path(root) / "apertures.npy", mmap_mode="r")
        z = np.empty(len(apertures), dtype=np.float64)
        pitch = np.empty(len(apertures), dtype=np.float64)
        for i, aperture in enumerate(apertures):
            d = support_width(np.asarray(aperture), config.input_pitch_m)
            z[i] = d * d / (config.wavelength_m * nf)
            pitch[i] = config.wavelength_m * z[i] / (config.grid * config.input_pitch_m)
        _GEOMETRY_CACHE[key] = (z, pitch)
    return _GEOMETRY_CACHE[key]


class GroupData(Dataset):
    """一个 NF 组的一个划分；条件为标准化后的 log(z), log(像素间距)。"""

    def __init__(self, root, group, split):
        self.root, self.group, self.split = Path(root), group, split
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete":
            raise ValueError("数据集尚未完整生成")
        self.config = PhysicsConfig(**self.manifest["config"])
        self.nf = next(g["nf"] for g in self.manifest["groups"] if g["name"] == group)
        self.apertures = np.load(self.root / "apertures.npy", mmap_mode="r")
        self.patterns = np.load(self.root / group / "patterns.npy", mmap_mode="r")
        with np.load(self.root / "splits.npz") as splits:
            self.ids = splits[split].copy()
            train_ids = splits["train"].copy()
        expected = (self.manifest["samples"], self.config.grid, self.config.grid)
        if self.apertures.shape != expected or self.patterns.shape != expected:
            raise ValueError("数据形状与 manifest 不一致")
        self.z_all, self.pitch_all = geometry_arrays(self.root, group)
        self.conditions = np.log(np.stack((self.z_all, self.pitch_all), axis=1))
        self.condition_mean = self.conditions[train_ids].mean(0)
        self.condition_std = np.maximum(self.conditions[train_ids].std(0), 1e-6)
        self._target_mean = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        i = int(self.ids[index])
        a = torch.from_numpy(np.array(self.apertures[i], copy=True)).unsqueeze(0)
        target = torch.from_numpy(np.array(self.patterns[i], copy=True)).unsqueeze(0)
        condition = torch.tensor(
            (self.conditions[i] - self.condition_mean) / self.condition_std,
            dtype=torch.float32)
        return a, target, condition, i

    def target_mean(self):
        # 只用训练划分的标签，避免验证/测试信息进入输出层初始化。
        if self._target_mean is None:
            train_ids = self.ids
            self._target_mean = float(np.mean(
                [self.patterns[int(i)].mean(dtype=np.float64) for i in train_ids]))
        return self._target_mean

    def z_tensor(self, ids, device):
        return torch.tensor([self.z_all[int(i)] for i in ids],
                            dtype=torch.float64, device=device)


# ============================================================
# 3. 训练与评估
# ============================================================
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用的 CUDA")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    return {key: float(np.mean([row[key] for row in rows]))
            for key in ("mae", "rmse", "psnr_db", "corr")}


def epoch_pass(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    totals = {k: 0.0 for k in ("total", "l1", "freq", "ssim", "res")}
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for a, target, condition, ids in loader:
            a, target, condition = a.to(device), target.to(device), condition.to(device)
            z = loader.dataset.z_tensor(ids, device)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            pred, _, delta = model(a, condition, z)
            loss, stats = criterion(pred, target, delta)
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失出现非有限值")
            if optimizer:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            totals["total"] += loss.item() * len(a)
            for key, value in stats.items():
                totals[key] += value * len(a)
            count += len(a)
    return {k: v / count for k, v in totals.items()}


@torch.no_grad()
def score(model, loader, device):
    model.eval()
    rows = []
    for a, target, condition, ids in loader:
        pred = model(a.to(device), condition.to(device), loader.dataset.z_tensor(ids, device))[0]
        for ident, p, t in zip(ids.tolist(), pred[:, 0].cpu().numpy(), target[:, 0].numpy()):
            rows.append({"id": ident, **image_metrics(p, t)})
    return aggregate(rows)


def checkpoint_payload(model, args, dataset, kind, propagator, epoch, val_l1, nf):
    return {"state_dict": model.state_dict(), "kind": kind, "base_ch": args.base_ch,
            "nf": nf, "epoch": epoch, "val_l1": val_l1,
            "condition_mean": dataset.condition_mean.tolist(),
            "condition_std": dataset.condition_std.tolist(),
            "physics_config": dataset.config.to_dict(),
            "propagator": propagator, "asm_precision": "float32"}


def plot_history(history, path, args, kind, nf, initial_l1):
    panels = [("l1", "L1 loss", "L1")]
    if args.lambda_freq > 0:
        panels.append(("freq", "Freq log-magnitude L1", "freq"))
    if args.lambda_ssim > 0:
        panels.append(("ssim", "SSIM loss", "1 - SSIM"))
    if kind == "physics" and args.lambda_res > 0:
        panels.append(("res", "Residual regularization", "|ΔI|"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    epochs = [row["epoch"] for row in history]
    for ax, (key, title, ylabel) in zip(axes[0], panels):
        for split in ("train", "val"):
            ax.semilogy(epochs, [max(row[f"{split}_{key}"], 1e-12) for row in history],
                        label=split, lw=1.5)
        if key == "l1":
            ax.axhline(max(initial_l1, 1e-12), ls=":", color="gray", label="initial val")
        ax.set(xlabel="epoch", ylabel=ylabel, title=title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"NF={nf:g}  {kind}")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_samples(model, dataset, device, path, kind, nf, n_show=4):
    """legacy 五列图：光阑 | 物理层 | 预测 | 真值 | 残差。"""
    model.eval()
    rng = np.random.RandomState(20261021)
    count = min(n_show, len(dataset))
    indices = rng.choice(len(dataset), count, replace=False)
    columns = 5 if kind == "physics" else 4
    fig, axes = plt.subplots(count, columns, figsize=(4 * columns, 4 * count), squeeze=False)
    for row, index in enumerate(indices):
        a, target, condition, ident = dataset[int(index)]
        a = a.unsqueeze(0).to(device)
        condition = condition.unsqueeze(0).to(device)
        z = dataset.z_tensor([int(ident)], device)
        target_np = target[0].numpy()
        if kind == "physics":
            predicted, simulated, delta = model(a, condition, z)
            predicted_np = predicted[0, 0].cpu().numpy()
            simulated = simulated[0, 0].cpu().numpy()
            values = [a[0, 0].cpu().numpy(), simulated, predicted_np, target_np,
                      delta[0, 0].cpu().numpy()]
            titles = ["Aperture", "I_sim (physics)", "I_pred (physics+res)",
                      "Ground truth", "Residual ΔI"]
            note = (f"sim MAE={np.mean(abs(simulated - target_np)):.5f}  "
                    f"pred MAE={np.mean(abs(predicted_np - target_np)):.5f}")
        else:
            predicted = model(a, condition, z)[0]
            predicted_np = predicted[0, 0].cpu().numpy()
            values = [a[0, 0].cpu().numpy(), predicted_np, target_np,
                      np.abs(predicted_np - target_np)]
            titles = ["Aperture", "I_pred (pure U-Net)", "Ground truth", "Absolute error"]
            note = f"pred MAE={np.mean(abs(predicted_np - target_np)):.5f}"
        for col, (value, title) in enumerate(zip(values, titles)):
            ax = axes[row, col]
            if col == 0:
                image = ax.imshow(value, cmap="gray", vmin=0, vmax=1)
            elif title == "Residual ΔI":
                limit = float(np.abs(value).max()) + 1e-12
                image = ax.imshow(value, cmap="RdBu_r", vmin=-limit, vmax=limit)
            elif title == "Absolute error":
                image = ax.imshow(value, cmap="inferno")
            else:
                image = ax.imshow(value, cmap="inferno", vmin=0, vmax=1)
            if row == 0:
                ax.set_title(title, fontsize=12)
            ax.axis("off")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        axes[row, -1].set_xlabel(note, fontsize=9)
    fig.suptitle(f"NF={nf:g} | validation samples", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_group(root, group, args, device, out_root):
    name, nf = group["name"], group["nf"]
    sets = {split: GroupData(root, name, split) for split in ("train", "val", "test")}
    loaders = {split: DataLoader(value, batch_size=args.batch_size,
                                 shuffle=(split == "train"),
                                 generator=torch.Generator().manual_seed(args.seed) if split == "train" else None,
                                 num_workers=args.workers)
               for split, value in sets.items()}
    propagator = ("fraunhofer" if nf < sets["train"].config.far_threshold else "asm") \
        if args.propagator == "hybrid" else args.propagator
    out = out_root / name
    out.mkdir()

    rows = []
    for kind in args.models:
        started = time.perf_counter()
        folder = out / kind
        folder.mkdir()
        seed_all(args.seed)
        model = ForwardModel(kind, args.base_ch, sets["train"].config.log_alpha,
                             sets["train"].target_mean(), propagator,
                             sets["train"].config).to(device)
        criterion = Loss(lambda_freq=args.lambda_freq, lambda_ssim=args.lambda_ssim,
                         lambda_res=args.lambda_res if kind == "physics" else 0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        initial = epoch_pass(model, loaders["val"], criterion, device)
        best = initial["l1"]
        torch.save(checkpoint_payload(model, args, sets["train"], kind, propagator,
                                      0, best, nf), folder / "best_model.pt")
        print(f"NF={nf:g} {kind} epoch 0: val_loss={best:.4e}", flush=True)

        history = []
        for epoch in range(args.epochs):
            train_stats = epoch_pass(model, loaders["train"], criterion, device, optimizer)
            val_stats = epoch_pass(model, loaders["val"], criterion, device)
            scheduler.step()
            row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
                   **{"train_" + k: v for k, v in train_stats.items()},
                   **{"val_" + k: v for k, v in val_stats.items()}}
            history.append(row)
            write_csv(folder / "history.csv", history)
            if val_stats["l1"] < best:
                best = val_stats["l1"]
                torch.save(checkpoint_payload(model, args, sets["train"], kind,
                                              propagator, epoch + 1, best, nf),
                           folder / "best_model.pt")
            print(f"NF={nf:g} {kind} epoch {epoch + 1}/{args.epochs}: "
                  f"train_loss={train_stats['l1']:.4e} val_loss={val_stats['l1']:.4e}",
                  flush=True)

        checkpoint = torch.load(folder / "best_model.pt", map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        test = score(model, loaders["test"], device)
        with (folder / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(test, f, ensure_ascii=False, indent=2)
        with (folder / "selection.json").open("w", encoding="utf-8") as f:
            json.dump({"best_epoch": checkpoint["epoch"],
                       "best_validation_l1": checkpoint["val_l1"],
                       "initial_validation_l1": initial["l1"],
                       "test": test,
                       "wall_seconds": time.perf_counter() - started},
                      f, ensure_ascii=False, indent=2)
        plot_history(history, folder / "loss_curve.png", args, kind, nf, initial["l1"])
        plot_samples(model, sets["val"], device, folder / "train_result.png",
                     kind, nf)
        rows.append({"nf": nf, "model": kind, "propagator": propagator,
                     "best_epoch": checkpoint["epoch"],
                     "best_val_l1": checkpoint["val_l1"], **test})
        print(f"NF={nf:g} {kind} 完成：best val={best:.4e}, test MAE={test['mae']:.4e} | {folder}",
              flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/v1", help="generate.py 的输出目录")
    parser.add_argument("--out", default=None, help="新目录；默认 runs/<时间戳>")
    parser.add_argument("--groups", type=float, nargs="+", help="只训练指定 NF 组")
    parser.add_argument("--models", nargs="+", choices=("physics", "baseline"),
                        default=("physics",),
                        help="默认只训练 physics；需要纯网络对照时显式加 baseline")
    parser.add_argument("--propagator", choices=("fraunhofer", "asm", "hybrid"),
                        default="fraunhofer",
                        help="物理层；hybrid 在 NF<切换点用 FFT，其余用 ASM")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-freq", type=float, default=0.1)
    parser.add_argument("--lambda-ssim", type=float, default=0.0,
                        help="内置 SSIM，无需额外依赖")
    parser.add_argument("--lambda-res", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0, help="Windows 下建议 0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.base_ch < 8 or args.base_ch % 8:
        parser.error("epochs/batch-size 必须为正，base-ch 必须是 8 的倍数")
    if args.lr <= 0 or min(args.lambda_freq, args.lambda_ssim, args.lambda_res) < 0:
        parser.error("学习率必须为正，损失系数不能为负")

    root = Path(args.data).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise SystemExit("数据集不完整，请先运行 generate.py")
    groups = manifest["groups"]
    if args.groups:
        wanted = set(float(x) for x in args.groups)
        groups = [g for g in groups if g["nf"] in wanted]
        if not groups:
            parser.error("--groups 指定的 NF 不在数据集中")
    if not groups:
        raise SystemExit("没有可训练的 NF 组")
    for split in ("train", "val", "test"):
        if not manifest["split_counts"][split]:
            raise SystemExit(f"{split} 划分为空，请增加 --samples")

    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    out = Path(args.out) if args.out else Path("runs") / time.strftime("%Y%m%d_%H%M%S")
    out = out.resolve()
    if out.exists():
        raise SystemExit(f"输出目录已存在：{out}；请换一个新名字")
    out.mkdir(parents=True)
    with (out / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({"arguments": vars(args), "device": str(device),
                   "data": str(root), "groups": [g["nf"] for g in groups]},
                  f, ensure_ascii=False, indent=2)

    print(f"设备：{device}；数据：{root}；输出：{out}", flush=True)
    rows = []
    for group in groups:
        rows.extend(run_group(root, group, args, device, out))
        write_csv(out / "summary.csv", rows)
    print(f"训练完成：{out}")


if __name__ == "__main__":
    main()
