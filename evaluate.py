"""
evaluate.py

推理验证脚本。

用法:
    python evaluate.py                        # 用 data/ 下的数据
    python evaluate.py --model best_model.pt  # 指定模型
    python evaluate.py --data_dir data        # 指定数据目录

输出:
    eval_results/
    ├── samples.png            # 8 个样本的对比可视化
    ├── error_map.png          # 误差图
    ├── metrics.txt            # 定量指标
    ├── error_histogram.png    # 误差分布直方图
    ├── extreme_test.png       # 极端测试（单缝、圆孔、全通、全挡）
    └── sweep_*.png            # 参数扫描图
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 从 train.py 导入模型定义
from train import PhysicsGuidedModel, DiffractionDataset, Loss


# ============================================================
# 参数
# ============================================================
DEFAULT_MODEL = 'best_model.pt'
DEFAULT_DATA_DIR = 'data'
OUTPUT_DIR = 'eval_results'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# 工具函数
# ============================================================
def load_model(model_path, device=DEVICE):
    """加载模型并设为推理模式"""
    model = PhysicsGuidedModel().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ 模型加载: {model_path}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model


def compute_metrics(I_pred, I_target):
    """
    计算多种指标。
    
    Args:
        I_pred, I_target: (H, W) numpy array
    Returns:
        dict of metrics
    """
    diff = I_pred - I_target
    abs_diff = np.abs(diff)
    
    mse = np.mean(diff ** 2)
    mae = np.mean(abs_diff)
    rmse = np.sqrt(mse)
    
    # PSNR
    max_val = max(I_target.max(), I_pred.max()) + 1e-12
    psnr = 20 * np.log10(max_val / (rmse + 1e-12))
    
    # 相对误差
    rel_err = np.sum(abs_diff) / (np.sum(np.abs(I_target)) + 1e-12)
    
    # 相关系数
    if I_pred.std() > 1e-9 and I_target.std() > 1e-9:
        corr = np.corrcoef(I_pred.flatten(), I_target.flatten())[0, 1]
    else:
        corr = 0.0
    
    # 峰值位置误差（以最大值位置为准）
    peak_pred = np.unravel_index(np.argmax(I_pred), I_pred.shape)
    peak_target = np.unravel_index(np.argmax(I_target), I_target.shape)
    peak_dist = np.sqrt((peak_pred[0] - peak_target[0])**2 +
                        (peak_pred[1] - peak_target[1])**2)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'PSNR': psnr,
        'RelErr': rel_err,
        'Corr': corr,
        'PeakDist': peak_dist,
    }


# ============================================================
# 1. 批量推理 + 指标统计
# ============================================================
@torch.no_grad()
def evaluate_dataset(model, dataset, device=DEVICE):
    """
    对整个数据集推理，返回所有预测和指标。
    """
    print(f"\n推理 {len(dataset)} 个样本...")
    
    all_metrics = {
        'sim': [], 'pred': [], 'impr': []
    }
    samples = []  # 保存一些样本用于可视化
    
    for i in range(len(dataset)):
        a, p = dataset[i]
        a_t = a.unsqueeze(0).to(device)
        p_np = p.squeeze().numpy()
        
        I_pred, I_sim, delta = model(a_t)
        I_pred = I_pred.squeeze().cpu().numpy()
        I_sim = I_sim.squeeze().cpu().numpy()
        a_np = a.squeeze().numpy()
        
        # 指标
        m_sim = compute_metrics(I_sim, p_np)
        m_pred = compute_metrics(I_pred, p_np)
        
        all_metrics['sim'].append(m_sim['MAE'])
        all_metrics['pred'].append(m_pred['MAE'])
        all_metrics['impr'].append(
            (m_sim['MAE'] - m_pred['MAE']) / (m_sim['MAE'] + 1e-12) * 100
        )
        
        # 保存前 8 个样本用于可视化
        if i < 8:
            samples.append({
                'idx': i,
                'aperture': a_np,
                'I_sim': I_sim,
                'I_pred': I_pred,
                'I_target': p_np,
                'delta': delta.squeeze().cpu().numpy(),
                'metrics_sim': m_sim,
                'metrics_pred': m_pred,
            })
    
    # 汇总
    summary = {
        'MAE_sim': np.mean(all_metrics['sim']),
        'MAE_pred': np.mean(all_metrics['pred']),
        'MAE_sim_std': np.std(all_metrics['sim']),
        'MAE_pred_std': np.std(all_metrics['pred']),
        'Improvement': np.mean(all_metrics['impr']),
        'Improvement_std': np.std(all_metrics['impr']),
    }
    
    return samples, all_metrics, summary


# ============================================================
# 2. 可视化: 多样本对比
# ============================================================
def plot_samples(samples, save_path):
    """
    8 个样本 × 5 列: 光阑 | 物理层 | 预测 | 真值 | 残差
    """
    n = len(samples)
    fig = plt.figure(figsize=(20, 4 * n))
    gs = GridSpec(n, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.05])
    
    col_titles = ['Aperture', 'I_sim (physics)', 'I_pred (physics+res)',
                  'Ground truth', 'Residual ΔI', 'MAE']
    
    for row, s in enumerate(samples):
        # 光阑
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(s['aperture'], cmap='gray')
        if row == 0:
            ax.set_title(col_titles[0], fontsize=11)
        ax.axis('off')
        
        # 物理层
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(s['I_sim'], cmap='inferno')
        if row == 0:
            ax.set_title(col_titles[1], fontsize=11)
        ax.axis('off')
        
        # 预测
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(s['I_pred'], cmap='inferno')
        if row == 0:
            ax.set_title(col_titles[2], fontsize=11)
        ax.axis('off')
        
        # 真值
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(s['I_target'], cmap='inferno')
        if row == 0:
            ax.set_title(col_titles[3], fontsize=11)
        ax.axis('off')
        
        # 残差
        ax = fig.add_subplot(gs[row, 4])
        vmax = np.abs(s['delta']).max() + 1e-12
        im = ax.imshow(s['delta'], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        if row == 0:
            ax.set_title(col_titles[4], fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # MAE 文字
        ax = fig.add_subplot(gs[row, 5])
        ax.axis('off')
        text = (f"sim:  {s['metrics_sim']['MAE']:.4f}\n"
                f"pred: {s['metrics_pred']['MAE']:.4f}")
        ax.text(0.1, 0.5, text, fontsize=9, va='center',
                family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ============================================================
# 3. 可视化: 误差图
# ============================================================
def plot_error_map(samples, save_path):
    """
    选取一个样本，展示各种误差图。
    """
    s = samples[0]
    
    err_sim = np.abs(s['I_sim'] - s['I_target'])
    err_pred = np.abs(s['I_pred'] - s['I_target'])
    improvement = err_sim - err_pred
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    
    # 第一行
    axes[0][0].imshow(s['aperture'], cmap='gray')
    axes[0][0].set_title('Aperture')
    
    axes[0][1].imshow(s['I_sim'], cmap='inferno')
    axes[0][1].set_title('I_sim')
    
    axes[0][2].imshow(s['I_pred'], cmap='inferno')
    axes[0][2].set_title('I_pred')
    
    axes[0][3].imshow(s['I_target'], cmap='inferno')
    axes[0][3].set_title('Ground truth')
    
    # 第二行: 误差
    vmax_err = max(err_sim.max(), err_pred.max()) + 1e-12
    
    im1 = axes[1][0].imshow(err_sim, cmap='hot', vmin=0, vmax=vmax_err)
    axes[1][0].set_title(f'|Error| physics\nMAE={err_sim.mean():.4f}')
    plt.colorbar(im1, ax=axes[1][0], fraction=0.046)
    
    im2 = axes[1][1].imshow(err_pred, cmap='hot', vmin=0, vmax=vmax_err)
    axes[1][1].set_title(f'|Error| predicted\nMAE={err_pred.mean():.4f}')
    plt.colorbar(im2, ax=axes[1][1], fraction=0.046)
    
    vmax_imp = np.abs(improvement).max() + 1e-12
    im3 = axes[1][2].imshow(improvement, cmap='RdYlGn',
                             vmin=-vmax_imp, vmax=vmax_imp)
    axes[1][2].set_title('Improvement\n(red=worse, green=better)')
    plt.colorbar(im3, ax=axes[1][2], fraction=0.046)
    
    # 中心行剖面
    H = s['I_sim'].shape[0]
    axes[1][3].plot(s['I_sim'][H//2, :], label='physics', lw=1)
    axes[1][3].plot(s['I_pred'][H//2, :], label='predicted', lw=1)
    axes[1][3].plot(s['I_target'][H//2, :], '--', label='truth', lw=1)
    axes[1][3].set_title('Center row profile')
    axes[1][3].legend(fontsize=8)
    axes[1][3].grid(True, alpha=0.3)
    
    for ax in axes[0]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ============================================================
# 4. 可视化: 误差直方图
# ============================================================
def plot_error_histogram(all_metrics, save_path):
    """
    每个样本的 MAE 分布。
    """
    sim = np.array(all_metrics['sim'])
    pred = np.array(all_metrics['pred'])
    impr = np.array(all_metrics['impr'])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # MAE 直方图
    axes[0].hist(sim, bins=30, alpha=0.6, label=f'physics (μ={sim.mean():.4f})')
    axes[0].hist(pred, bins=30, alpha=0.6, label=f'predicted (μ={pred.mean():.4f})')
    axes[0].set_xlabel('MAE')
    axes[0].set_ylabel('count')
    axes[0].set_title('MAE distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 改善百分比直方图
    axes[1].hist(impr, bins=30, color='green', alpha=0.7)
    axes[1].axvline(impr.mean(), color='red', linestyle='--',
                     label=f'mean={impr.mean():.1f}%')
    axes[1].axvline(0, color='black', linestyle='-', alpha=0.5)
    axes[1].set_xlabel('Improvement (%)')
    axes[1].set_ylabel('count')
    axes[1].set_title('Improvement by predicted over physics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 散点: physics MAE vs predicted MAE
    axes[2].scatter(sim, pred, s=10, alpha=0.5)
    lim = max(sim.max(), pred.max()) * 1.05
    axes[2].plot([0, lim], [0, lim], 'r--', label='y=x')
    axes[2].set_xlabel('Physics MAE')
    axes[2].set_ylabel('Predicted MAE')
    axes[2].set_title('Per-sample comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ============================================================
# 5. 极端测试
# ============================================================
@torch.no_grad()
def extreme_test(model, device=DEVICE, save_path='eval_results/extreme_test.png'):
    """
    用极端情况测试模型。
    """
    print("\n极端测试...")
    
    N = 256
    tests = {}
    
    # 单缝
    a = np.zeros((N, N), dtype=np.float32)
    a[40:N-40, N//2-4:N//2+4] = 1.0
    tests['Single slit'] = a
    
    # 双缝
    a = np.zeros((N, N), dtype=np.float32)
    a[40:N-40, N//2-30:N//2-22] = 1.0
    a[40:N-40, N//2+22:N//2+30] = 1.0
    tests['Double slit'] = a
    
    # 圆孔
    a = np.zeros((N, N), dtype=np.float32)
    yy, xx = np.ogrid[:N, :N]
    a[(xx - N//2)**2 + (yy - N//2)**2 <= 30**2] = 1.0
    tests['Circle'] = a
    
    # 方孔
    a = np.zeros((N, N), dtype=np.float32)
    a[N//2-30:N//2+30, N//2-30:N//2+30] = 1.0
    tests['Square'] = a
    
    # 光栅
    a = np.zeros((N, N), dtype=np.float32)
    for x in range(0, N, 16):
        a[:, x:x+6] = 1.0
    tests['Grating'] = a
    
    # 全通
    tests['Full open'] = np.ones((N, N), dtype=np.float32)
    
    # 全挡
    tests['Full blocked'] = np.zeros((N, N), dtype=np.float32)
    
    # 单个点
    a = np.zeros((N, N), dtype=np.float32)
    a[N//2, N//2] = 1.0
    tests['Single point'] = a
    
    n = len(tests)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    
    for i, (name, a) in enumerate(tests.items()):
        a_t = torch.tensor(a).unsqueeze(0).unsqueeze(0).to(device)
        I_pred, I_sim, delta = model(a_t)
        I_pred = I_pred.squeeze().cpu().numpy()
        I_sim = I_sim.squeeze().cpu().numpy()
        
        axes[i][0].imshow(a, cmap='gray')
        axes[i][0].set_title(f'{name}')
        axes[i][0].axis('off')
        
        axes[i][1].imshow(I_sim, cmap='inferno')
        axes[i][1].set_title('physics only')
        axes[i][1].axis('off')
        
        axes[i][2].imshow(I_pred, cmap='inferno')
        axes[i][2].set_title('predicted')
        axes[i][2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ============================================================
# 6. 参数扫描
# ============================================================
@torch.no_grad()
def sweep_slit_width(model, device=DEVICE,
                     save_path='eval_results/sweep_slit_width.png'):
    """扫描缝宽"""
    print("\n参数扫描: 缝宽...")
    
    N = 256
    widths = [2, 4, 6, 8, 12, 16, 24, 32]
    
    fig, axes = plt.subplots(2, len(widths), figsize=(3 * len(widths), 6))
    
    for i, w in enumerate(widths):
        a = np.zeros((N, N), dtype=np.float32)
        a[40:N-40, N//2-w//2:N//2+w//2] = 1.0
        
        a_t = torch.tensor(a).unsqueeze(0).unsqueeze(0).to(device)
        I_pred, _, _ = model(a_t)
        I_pred = I_pred.squeeze().cpu().numpy()
        
        axes[0][i].imshow(I_pred, cmap='inferno')
        axes[0][i].set_title(f'w={w}')
        axes[0][i].axis('off')
        
        axes[1][i].plot(I_pred[N//2, :], lw=0.8)
        axes[1][i].set_xlim(0, N)
        axes[1][i].set_ylim(0, 1.05)
        axes[1][i].grid(True, alpha=0.3)
        axes[1][i].set_title(f'profile w={w}', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


@torch.no_grad()
def sweep_circle_radius(model, device=DEVICE,
                        save_path='eval_results/sweep_circle_radius.png'):
    """扫描圆孔半径"""
    print("\n参数扫描: 圆孔半径...")
    
    N = 256
    radii = [5, 10, 15, 20, 30, 40, 55, 70]
    
    fig, axes = plt.subplots(2, len(radii), figsize=(3 * len(radii), 6))
    yy, xx = np.ogrid[:N, :N]
    
    for i, r in enumerate(radii):
        a = ((xx - N//2)**2 + (yy - N//2)**2 <= r**2).astype(np.float32)
        
        a_t = torch.tensor(a).unsqueeze(0).unsqueeze(0).to(device)
        I_pred, _, _ = model(a_t)
        I_pred = I_pred.squeeze().cpu().numpy()
        
        axes[0][i].imshow(I_pred, cmap='inferno')
        axes[0][i].set_title(f'r={r}')
        axes[0][i].axis('off')
        
        axes[1][i].plot(I_pred[N//2, :], lw=0.8)
        axes[1][i].set_xlim(0, N)
        axes[1][i].set_ylim(0, 1.05)
        axes[1][i].grid(True, alpha=0.3)
        axes[1][i].set_title(f'profile r={r}', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


@torch.no_grad()
def sweep_grating_period(model, device=DEVICE,
                          save_path='eval_results/sweep_grating_period.png'):
    """扫描光栅周期"""
    print("\n参数扫描: 光栅周期...")
    
    N = 256
    periods = [6, 10, 14, 18, 24, 32, 48, 64]
    
    fig, axes = plt.subplots(2, len(periods), figsize=(3 * len(periods), 6))
    
    for i, p in enumerate(periods):
        a = np.zeros((N, N), dtype=np.float32)
        bar = max(2, p // 2)
        for x in range(0, N, p):
            a[:, x:x+bar] = 1.0
        
        a_t = torch.tensor(a).unsqueeze(0).unsqueeze(0).to(device)
        I_pred, _, _ = model(a_t)
        I_pred = I_pred.squeeze().cpu().numpy()
        
        axes[0][i].imshow(I_pred, cmap='inferno')
        axes[0][i].set_title(f'p={p}')
        axes[0][i].axis('off')
        
        axes[1][i].plot(I_pred[N//2, :], lw=0.8)
        axes[1][i].set_xlim(0, N)
        axes[1][i].set_ylim(0, 1.05)
        axes[1][i].grid(True, alpha=0.3)
        axes[1][i].set_title(f'profile p={p}', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ============================================================
# 7. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--n_vis', type=int, default=8,
                        help='可视化样本数')
    parser.add_argument('--skip_sweep', action='store_true',
                        help='跳过参数扫描')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("推理验证")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"模型: {args.model}")
    print(f"数据: {args.data_dir}")
    print()
    
    # ---------- 加载模型 ----------
    model = load_model(args.model, DEVICE)
    
    # ---------- 加载数据 ----------
    ap_path = os.path.join(args.data_dir, 'apertures.npy')
    pat_path = os.path.join(args.data_dir, 'patterns.npy')
    
    if not os.path.exists(ap_path):
        raise FileNotFoundError(f"找不到 {ap_path}")
    
    dataset = DiffractionDataset(ap_path, pat_path)
    
    # 用最后 10% 作为验证集（和训练时一致）
    n = len(dataset)
    n_val = max(1, int(n * 0.1))
    _, val_set = torch.utils.data.random_split(
        dataset, [n - n_val, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"验证集大小: {len(val_set)}")
    
    # 如果验证集太大，只取前 500 个加速
    if len(val_set) > 500:
        indices = torch.arange(500)
        val_set = torch.utils.data.Subset(val_set, indices)
        print(f"  取前 500 个用于评估")
    
    # ---------- 批量推理 ----------
    samples, all_metrics, summary = evaluate_dataset(model, val_set, DEVICE)
    
    # ---------- 打印指标 ----------
    print("\n" + "=" * 60)
    print("定量指标")
    print("=" * 60)
    print(f"物理层 MAE:   {summary['MAE_sim']:.6f} ± {summary['MAE_sim_std']:.6f}")
    print(f"预测   MAE:   {summary['MAE_pred']:.6f} ± {summary['MAE_pred_std']:.6f}")
    print(f"平均改善:     {summary['Improvement']:+.2f}% ± {summary['Improvement_std']:.2f}%")
    
    # 保存到文件
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Data: {args.data_dir}\n")
        f.write(f"Val samples: {len(val_set)}\n\n")
        f.write(f"Physics MAE: {summary['MAE_sim']:.6f} ± {summary['MAE_sim_std']:.6f}\n")
        f.write(f"Predicted MAE: {summary['MAE_pred']:.6f} ± {summary['MAE_pred_std']:.6f}\n")
        f.write(f"Improvement: {summary['Improvement']:+.2f}% ± {summary['Improvement_std']:.2f}%\n")
    print(f"  → {args.output_dir}/metrics.txt")
    
    # ---------- 可视化 ----------
    print("\n生成可视化...")
    plot_samples(samples[:args.n_vis],
                 os.path.join(args.output_dir, 'samples.png'))
    plot_error_map(samples,
                    os.path.join(args.output_dir, 'error_map.png'))
    plot_error_histogram(all_metrics,
                          os.path.join(args.output_dir, 'error_histogram.png'))
    
    # ---------- 极端测试 ----------
    extreme_test(model, DEVICE,
                 os.path.join(args.output_dir, 'extreme_test.png'))
    
    # ---------- 参数扫描 ----------
    if not args.skip_sweep:
        sweep_slit_width(model, DEVICE,
                         os.path.join(args.output_dir, 'sweep_slit_width.png'))
        sweep_circle_radius(model, DEVICE,
                            os.path.join(args.output_dir, 'sweep_circle_radius.png'))
        sweep_grating_period(model, DEVICE,
                              os.path.join(args.output_dir, 'sweep_grating_period.png'))
    
    print("\n" + "=" * 60)
    print(f"✓ 完成！结果保存在: {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()