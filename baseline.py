"""
baseline.py (修复版)

无物理信息基线模型，用于和 PhysicsGuidedModel 做对比实验。

关键修复 (相比原版):
    1. PureUNet 去掉 sigmoid，改用线性输出
       → 真值均值≈0.05，sigmoid 初始输出≈0.5，会让 L1 从 0.45 起步并陷入饱和
    2. 输出层小权重 + bias=target_mean 初始化
       → 训练起点 = 输出均值，L1≈0.05，而非 0.45
    3. 训练函数适配新版 Loss (返回 stats dict)
    4. baseline 显式关闭 SSIM / 残差正则，避免污染 L1 指标
    5. LR 从 1e-3 降到 3e-4，baseline 起点差，大 LR 会发散
    6. 训练前打印初始 L1，作为“修复是否生效”的自检

设计原则 (不变):
    1. 与 PhysicsGuidedModel 的残差 U-Net 使用完全相同的骨干
    2. 参数量几乎一致 (差 288 个, 忽略不计)
    3. 唯一区别: 无 FFT 物理层, 端到端从光阑直接预测衍射图样

用法:
    python baseline.py train     # 训练基线模型
    python baseline.py compare   # 与 PhysicsGuidedModel 全面对比
    python baseline.py all       # 训练 + 对比
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

# 复用 train.py 的所有组件，保证配置一致
from train import (
    DiffractionDataset, Loss, PhysicsGuidedModel,
    BATCH_SIZE, EPOCHS, LR, VAL_RATIO, DEVICE,
    DATA_DIR, LOG_ALPHA, SAVE_PATH as PHYSICS_PATH,
)


BASELINE_PATH = 'baseline_model.pt'
COMPARE_DIR = 'comparison'

# baseline 专用 LR (起点差, 大 LR 会震荡发散)
BASELINE_LR = 3e-4


# ============================================================
# 1. 基线模型 (PureUNet)
# ============================================================
class ConvBlock(nn.Module):
    """和 train.py 完全一致的卷积块"""
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


class PureUNet(nn.Module):
    """
    纯数据驱动 U-Net, 无任何物理先验。

    输入:  光阑  (B, 1, H, W)
    输出:  衍射  (B, 1, H, W), 线性输出 (不 clamp, 让损失约束)

    与 PhysicsGuidedModel 的残差网络相比:
        编码器 enc1: 1->32   (物理版: 2->32)
        其余层完全相同
        参数差: 9*32 = 288 个 (差异 < 0.02%)

    关键修复:
        1. 去掉 sigmoid → 避免在真值均值≈0.05 时陷入饱和区
        2. out_conv 权重小初始化 + bias=target_mean
           → 训练起点就是 "输出均值", L1≈0.05, 而非 0.45
    """

    def __init__(self, base_ch=32, target_mean=0.05):
        super().__init__()

        # 编码器 (与物理模型残差网络结构完全一致)
        self.enc1 = ConvBlock(1, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        # 解码器
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        # 输出层 (线性, 无 sigmoid)
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

        # ⚠️ 关键修复: 输出层初始化
        #    权重很小 → 初始输出主要由 bias 决定
        #    bias = target_mean → 初始 I_pred ≈ 0.05 (真值均值)
        nn.init.normal_(self.out_conv.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.out_conv.bias, target_mean)

        self.pool = nn.MaxPool2d(2)

    def forward(self, a):
        e1 = self.enc1(a)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        d3 = self.up3(b);  d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3); d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # ⚠️ 线性输出, 无 sigmoid
        return self.out_conv(d1)


# ============================================================
# 2. 训练基线
# ============================================================
def train_one_epoch_baseline(model, loader, optimizer, criterion, device):
    """一个 epoch 训练。返回聚合后的 {total, l1, freq} 字典。"""
    model.train()
    agg = {'total': 0.0, 'l1': 0.0, 'freq': 0.0}
    n = 0
    for a, p in loader:
        a, p = a.to(device), p.to(device)

        optimizer.zero_grad()
        I_pred = model(a)                                # 端到端, 无 I_sim
        loss, stats = criterion(I_pred, p, delta=None)   # 基线无 delta
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        agg['total'] += loss.item() * a.size(0)
        agg['l1']    += stats['l1'] * a.size(0)
        agg['freq']  += stats['freq'] * a.size(0)
        n += a.size(0)
    return {k: v / n for k, v in agg.items()}


@torch.no_grad()
def evaluate_baseline(model, loader, criterion, device):
    """验证。返回聚合后的 {total, l1, freq} 字典。"""
    model.eval()
    agg = {'total': 0.0, 'l1': 0.0, 'freq': 0.0}
    n = 0
    for a, p in loader:
        a, p = a.to(device), p.to(device)
        I_pred = model(a)
        loss, stats = criterion(I_pred, p, delta=None)
        agg['total'] += loss.item() * a.size(0)
        agg['l1']    += stats['l1'] * a.size(0)
        agg['freq']  += stats['freq'] * a.size(0)
        n += a.size(0)
    return {k: v / n for k, v in agg.items()}


def build_loaders():
    ap_path = os.path.join(DATA_DIR, 'apertures.npy')
    pat_path = os.path.join(DATA_DIR, 'patterns.npy')
    dataset = DiffractionDataset(ap_path, pat_path)

    n = len(dataset)
    n_val = int(n * VAL_RATIO)
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2)
    return dataset, train_set, val_set, train_loader, val_loader


def train():
    print("=" * 60)
    print("训练基线模型 (PureUNet, 无物理先验)")
    print("=" * 60)

    dataset, train_set, val_set, train_loader, val_loader = build_loaders()

    # ---------- 从数据算 target_mean, 用于输出层初始化 ----------
    pats = np.load(os.path.join(DATA_DIR, 'patterns.npy'))
    target_mean = float(pats.mean())
    print(f"目标均值: {target_mean:.5f}  (用于输出层初始化)")

    model = PureUNet(base_ch=32, target_mean=target_mean).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"基线参数量: {n_params:,}")

    # ---------- 前向自检: 初始 L1 应 ≈ target_mean 量级 ----------
    model.eval()
    with torch.no_grad():
        a0, p0 = dataset[0]
        I0 = model(a0.unsqueeze(0).to(DEVICE))
        l1_init = (I0 - p0.unsqueeze(0).to(DEVICE)).abs().mean().item()
    print(f"初始 L1 (训练前): {l1_init:.5f}")
    if l1_init > 0.3:
        print("⚠ 初始 L1 异常大! 检查 target_mean 是否正确, "
              "或 out_conv 初始化被后续代码覆盖")
    model.train()

    # ---------- 优化器 (baseline 专用 LR) ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASELINE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ---------- 损失: 显式关闭 SSIM / 残差正则 ----------
    # baseline 只关心数值拟合, SSIM 会和 L1 目标冲突并主导梯度
    criterion = Loss(
        lambda_freq=0.1,
        lambda_ssim=0.0,      # ← 关掉
        lambda_res=0.0,       # ← 关掉 (baseline 也没传 delta)
        use_log_freq=True,
    )
    print(f"损失配置: λ_freq={criterion.lambda_freq}, "
          f"λ_ssim={criterion.lambda_ssim}, "
          f"λ_res={criterion.lambda_res}, "
          f"log_freq={criterion.use_log_freq}")
    print(f"LR: {BASELINE_LR:.1e}")

    # ---------- 训练 ----------
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    best_val = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(EPOCHS):
        tr = train_one_epoch_baseline(model, train_loader, optimizer, criterion, DEVICE)
        va = evaluate_baseline(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history['train'].append(tr['l1'])
        history['val'].append(va['l1'])

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train L1: {tr['l1']:.5e} (freq {tr['freq']:.3e}) | "
              f"Val L1: {va['l1']:.5e}")

        if va['l1'] < best_val:
            best_val = va['l1']
            torch.save(model.state_dict(), BASELINE_PATH)

    print(f"\n基线最佳 Val L1: {best_val:.6e}")
    print(f"模型保存: {BASELINE_PATH}")

    # ---------- Loss 曲线 ----------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history['train'], label='train')
    ax.plot(history['val'], label='val')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title('Baseline training curve')
    ax.set_xlabel('epoch')
    ax.set_ylabel('L1')
    plt.tight_layout()
    plt.savefig('baseline_loss_curve.png', dpi=120)
    plt.close()
    print("Loss 曲线保存到: baseline_loss_curve.png")


# ============================================================
# 3. 对比评估
# ============================================================
def compute_metrics(I_pred, I_target):
    diff = I_pred - I_target
    abs_diff = np.abs(diff)
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(mse))
    max_val = max(I_target.max(), I_pred.max()) + 1e-12
    psnr = float(20 * np.log10(max_val / (rmse + 1e-12)))
    if I_pred.std() > 1e-9 and I_target.std() > 1e-9:
        corr = float(np.corrcoef(I_pred.flatten(), I_target.flatten())[0, 1])
    else:
        corr = 0.0
    return {'MAE': mae, 'RMSE': rmse, 'PSNR': psnr, 'Corr': corr}


def compare():
    print("=" * 60)
    print("对比: 物理引导 vs 纯数据驱动")
    print("=" * 60)

    os.makedirs(COMPARE_DIR, exist_ok=True)

    # ---------- 构建验证集 (与训练时同划分) ----------
    dataset, _, val_set, _, _ = build_loaders()
    if len(val_set) > 500:
        val_set = torch.utils.data.Subset(val_set, torch.arange(500))
    print(f"验证集: {len(val_set)} 个样本")

    # ---------- 加载两个模型 ----------
    physics = PhysicsGuidedModel(log_compress=True, alpha=LOG_ALPHA).to(DEVICE)
    physics.load_state_dict(torch.load(PHYSICS_PATH, map_location=DEVICE))
    physics.eval()

    # PureUNet 也要带 target_mean, 否则结构不一致
    pats = np.load(os.path.join(DATA_DIR, 'patterns.npy'))
    baseline = PureUNet(base_ch=32, target_mean=float(pats.mean())).to(DEVICE)
    baseline.load_state_dict(torch.load(BASELINE_PATH, map_location=DEVICE))
    baseline.eval()

    print(f"\n物理模型参数: {sum(p.numel() for p in physics.parameters()):,}")
    print(f"基线模型参数: {sum(p.numel() for p in baseline.parameters()):,}")

    # ---------- 逐样本评估 ----------
    metrics = {'physics_only': [], 'physics_pred': [], 'baseline': []}
    samples = []

    for i in range(len(val_set)):
        a, p = val_set[i]
        a_t = a.unsqueeze(0).to(DEVICE)
        p_np = p.squeeze().numpy()

        with torch.no_grad():
            I_pred_phys, I_sim, _ = physics(a_t)
            I_pred_base = baseline(a_t)

        I_sim_np   = I_sim.squeeze().cpu().numpy()
        I_phys_np  = I_pred_phys.squeeze().cpu().numpy()
        I_base_np  = I_pred_base.squeeze().cpu().numpy()
        a_np       = a.squeeze().numpy()

        m_sim  = compute_metrics(I_sim_np, p_np)
        m_phys = compute_metrics(I_phys_np, p_np)
        m_base = compute_metrics(I_base_np, p_np)

        metrics['physics_only'].append(m_sim)
        metrics['physics_pred'].append(m_phys)
        metrics['baseline'].append(m_base)

        if i < 8:
            samples.append({
                'aperture': a_np,
                'physics_only': I_sim_np,
                'physics_pred': I_phys_np,
                'baseline_pred': I_base_np,
                'target': p_np,
                'm_sim': m_sim, 'm_phys': m_phys, 'm_base': m_base,
            })

    # ---------- 汇总 ----------
    def agg(key):
        return {m: np.mean([x[m] for x in metrics[key]])
                for m in ('MAE', 'RMSE', 'PSNR', 'Corr')}

    s_sim  = agg('physics_only')
    s_phys = agg('physics_pred')
    s_base = agg('baseline')

    # ---------- 打印表格 ----------
    print("\n" + "=" * 78)
    print(f"{'Model':<28} | {'MAE':>10} | {'RMSE':>10} | {'PSNR':>8} | {'Corr':>8}")
    print("-" * 78)
    print(f"{'Physics only (no NN)':<28} | "
          f"{s_sim['MAE']:>10.6f} | {s_sim['RMSE']:>10.6f} | "
          f"{s_sim['PSNR']:>8.2f} | {s_sim['Corr']:>8.5f}")
    print(f"{'Baseline PureUNet':<28} | "
          f"{s_base['MAE']:>10.6f} | {s_base['RMSE']:>10.6f} | "
          f"{s_base['PSNR']:>8.2f} | {s_base['Corr']:>8.5f}")
    print(f"{'Physics + Residual (ours)':<28} | "
          f"{s_phys['MAE']:>10.6f} | {s_phys['RMSE']:>10.6f} | "
          f"{s_phys['PSNR']:>8.2f} | {s_phys['Corr']:>8.5f}")
    print("=" * 78)

    # ---------- 保存指标 ----------
    with open(os.path.join(COMPARE_DIR, 'metrics.txt'), 'w') as f:
        f.write(f"Val samples: {len(val_set)}\n\n")
        f.write(f"{'Model':<28} | {'MAE':>10} | {'RMSE':>10} | "
                f"{'PSNR':>8} | {'Corr':>8}\n")
        f.write("-" * 78 + "\n")
        for name, s in [('Physics only', s_sim),
                        ('Baseline PureUNet', s_base),
                        ('Physics + Residual', s_phys)]:
            f.write(f"{name:<28} | {s['MAE']:>10.6f} | {s['RMSE']:>10.6f} | "
                    f"{s['PSNR']:>8.2f} | {s['Corr']:>8.5f}\n")
        f.write("\nRelative MAE improvement over Baseline:\n")
        f.write(f"  Physics + Residual: "
                f"{(s_base['MAE'] - s_phys['MAE']) / s_base['MAE'] * 100:+.2f}%\n")

    # ---------- 可视化 ----------
    plot_side_by_side(samples, os.path.join(COMPARE_DIR, 'side_by_side.png'))
    plot_mae_distribution(metrics, os.path.join(COMPARE_DIR, 'mae_distribution.png'))
    plot_per_sample_mae(samples, metrics, os.path.join(COMPARE_DIR, 'per_sample.png'))

    print(f"\n✓ 对比结果保存到: {COMPARE_DIR}/")


def plot_side_by_side(samples, save_path):
    """6 列: 光阑 | 物理层 | 物理+残差 | 基线 | 真值 | 误差热图对比"""
    n = len(samples)
    fig = plt.figure(figsize=(24, 4 * n))
    gs = GridSpec(n, 6, figure=fig)

    titles = ['Aperture', 'Physics only', 'Physics + Residual (ours)',
              'Baseline PureUNet', 'Ground truth', '|Error| comparison']

    for row, s in enumerate(samples):
        # 光阑
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(s['aperture'], cmap='gray', vmin=0, vmax=1,
                  interpolation='nearest')
        if row == 0: ax.set_title(titles[0], fontsize=12)
        ax.axis('off')

        # 物理层
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(s['physics_only'], cmap='inferno', vmin=0, vmax=1,
                  interpolation='nearest')
        if row == 0: ax.set_title(titles[1], fontsize=12)
        ax.axis('off')
        ax.set_xlabel(f"MAE={s['m_sim']['MAE']:.4f}", fontsize=9)

        # 物理+残差
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(s['physics_pred'], cmap='inferno', vmin=0, vmax=1,
                  interpolation='nearest')
        if row == 0: ax.set_title(titles[2], fontsize=12, color='green')
        ax.axis('off')
        ax.set_xlabel(f"MAE={s['m_phys']['MAE']:.4f}", fontsize=9, color='green')

        # 基线
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(s['baseline_pred'], cmap='inferno', vmin=0, vmax=1,
                  interpolation='nearest')
        if row == 0: ax.set_title(titles[3], fontsize=12, color='red')
        ax.axis('off')
        ax.set_xlabel(f"MAE={s['m_base']['MAE']:.4f}", fontsize=9, color='red')

        # 真值
        ax = fig.add_subplot(gs[row, 4])
        ax.imshow(s['target'], cmap='inferno', vmin=0, vmax=1,
                  interpolation='nearest')
        if row == 0: ax.set_title(titles[4], fontsize=12)
        ax.axis('off')

        # 误差热图 (baseline_err - physics_err)
        ax = fig.add_subplot(gs[row, 5])
        err_base = np.abs(s['baseline_pred'] - s['target'])
        err_phys = np.abs(s['physics_pred'] - s['target'])
        diff = err_base - err_phys     # >0 物理更好
        vmax = np.abs(diff).max() + 1e-12
        im = ax.imshow(diff, cmap='RdYlGn', vmin=-vmax, vmax=vmax,
                       interpolation='nearest')
        if row == 0: ax.set_title(titles[5], fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


def plot_mae_distribution(metrics, save_path):
    """MAE 分布直方图对比"""
    m_sim  = np.array([x['MAE'] for x in metrics['physics_only']])
    m_phys = np.array([x['MAE'] for x in metrics['physics_pred']])
    m_base = np.array([x['MAE'] for x in metrics['baseline']])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    bins = np.linspace(0, max(m_base.max(), m_phys.max()), 40)
    axes[0].hist(m_base, bins=bins, alpha=0.6,
                 label=f'Baseline (μ={m_base.mean():.4f})',
                 color='red')
    axes[0].hist(m_phys, bins=bins, alpha=0.6,
                 label=f'Physics+Res (μ={m_phys.mean():.4f})',
                 color='green')
    axes[0].hist(m_sim, bins=bins, alpha=0.4,
                 label=f'Physics only (μ={m_sim.mean():.4f})',
                 color='blue')
    axes[0].set_xlabel('MAE'); axes[0].set_ylabel('count')
    axes[0].set_title('Per-sample MAE distribution')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # 散点: baseline vs physics
    axes[1].scatter(m_base, m_phys, s=12, alpha=0.5, c='green')
    lim = max(m_base.max(), m_phys.max()) * 1.05
    axes[1].plot([0, lim], [0, lim], 'k--', lw=1, label='y=x (equal)')
    axes[1].set_xlabel('Baseline MAE')
    axes[1].set_ylabel('Physics+Residual MAE')
    axes[1].set_title('Sample-wise comparison')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    better = (m_phys < m_base).mean() * 100
    axes[1].text(0.05, 0.95, f'Physics better on {better:.1f}% samples',
                 transform=axes[1].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', fc='lightgreen', alpha=0.6))

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


def plot_per_sample_mae(samples, metrics, save_path):
    """按 baseline MAE 排序，显示两种模型的相对表现"""
    m_phys = np.array([x['MAE'] for x in metrics['physics_pred']])
    m_base = np.array([x['MAE'] for x in metrics['baseline']])

    idx = np.argsort(m_base)
    m_base_sorted = m_base[idx]
    m_phys_sorted = m_phys[idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(idx))
    ax.plot(x, m_base_sorted, color='red', lw=1.5, label='Baseline')
    ax.plot(x, m_phys_sorted, color='green', lw=1.5, label='Physics + Residual')
    ax.set_xlabel('Sample (sorted by Baseline MAE)')
    ax.set_ylabel('MAE')
    ax.set_title('Per-sample MAE: Baseline vs Physics-guided')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ============================================================
# 4. 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs='?', default='all',
                        choices=['train', 'compare', 'all'])
    args = parser.parse_args()

    if args.mode in ('train', 'all'):
        train()
    if args.mode in ('compare', 'all'):
        compare()


if __name__ == '__main__':
    main()