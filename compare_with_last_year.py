"""
compare_with_last_year.py

目的
----
在不改动 train.py / generate_data.py 的前提下, 把去年 (pix2pix) 的工程拉过来,
在当前数据 (apertures.npy / patterns.npy) 上重新训练一个正向模型
(光阑 -> 衍射图样), 然后与今年 (物理引导) 的方案做直接对比。

相同条件
--------
- 数据: 与 generate_data.py 完全一致 (256x256 float32, [0,1])
- 划分: 与 train.py 使用同一个随机种子 (seed=42), 验证集样本一致
- 目标: log 压缩后的衍射强度 (LOG_ALPHA = 100)

对比维度
--------
1. 训练曲线对比 (L1 损失)
2. 验证集逐样本定量指标 (L1 / MSE / SSIM 风格对比)
3. 定性可视化 (光阑 | 真值 | 物理层 | 今年预测 | 去年预测 | 残差图)

运行
----
    python compare_with_last_year.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# 直接复用今年的数据类与模型
from train import DiffractionDataset, PhysicsGuidedModel

# ============================================================
# 配置
# ============================================================
DATA_DIR = 'data'
CURRENT_MODEL_PATH = 'best_model.pt'          # 由 train.py 生成
PIX2PIX_MODEL_PATH = 'pix2pix_last_year.pt'   # 本脚本训练
COMPARE_FIG        = 'compare_result.png'
CURVE_FIG          = 'compare_curve.png'

BATCH_SIZE = 8
EPOCHS     = 30          # pix2pix 收敛相对较快, 30 轮足够展示差距
LR         = 2e-4
LAMBDA_L1  = 100.0
VAL_RATIO  = 0.1
SEED       = 42
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_ALPHA  = 100.0       # 必须与 generate_data.py 一致


# ============================================================
# 1. 去年 pix2pix 的模块 (从 notebook 原样复制)
# ============================================================
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2,
                 padding=1, apply_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                            padding, bias=not apply_bn)]
        if apply_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2,
                 padding=1, dropout=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                     stride, padding, bias=False),
                  nn.BatchNorm2d(out_channels)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Pix2PixGenerator(nn.Module):
    """去年 pix2pix 的 U-Net 生成器 (输入/输出 3 通道, 原样保留)"""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = DownSample(in_channels, 64, apply_bn=False)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)
        self.down5 = DownSample(512, 512)
        self.down6 = DownSample(512, 512)

        self.up1 = UpSample(512, 512, dropout=True)
        self.up2 = UpSample(1024, 512, dropout=True)
        self.up3 = UpSample(1024, 256)
        self.up4 = UpSample(512, 128)
        self.up5 = UpSample(256, 64)

        self.final = nn.ConvTranspose2d(128, out_channels,
                                        kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1)
        d3 = self.down3(d2); d4 = self.down4(d3)
        d5 = self.down5(d4); d6 = self.down6(d5)

        u1 = self.up1(d6); u1 = torch.cat([u1, d5], dim=1)
        u2 = self.up2(u1); u2 = torch.cat([u2, d4], dim=1)
        u3 = self.up3(u2); u3 = torch.cat([u3, d3], dim=1)
        u4 = self.up4(u3); u4 = torch.cat([u4, d2], dim=1)
        u5 = self.up5(u4); u5 = torch.cat([u5, d1], dim=1)

        return self.tanh(self.final(u5))


class Pix2PixDiscriminator(nn.Module):
    """去年 pix2pix 的 PatchGAN 判别器"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            DownSample(64, 128),
            DownSample(128, 256),
            DownSample(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


# ============================================================
# 2. 单通道 <-> 3 通道 适配 (数据不变, 只改输入格式)
# ============================================================
def to_3ch(x):
    """(B,1,H,W) [0,1] -> (B,3,H,W) [-1,1]"""
    return x.repeat(1, 3, 1, 1) * 2 - 1


def from_3ch(x):
    """(B,3,H,W) [-1,1] -> (B,1,H,W) [0,1]"""
    return ((x[:, :1] + 1) / 2).clamp(0, 1)


# ============================================================
# 3. 数据加载 (与 train.py 完全相同的划分)
# ============================================================
def load_split():
    ap = os.path.join(DATA_DIR, 'apertures.npy')
    pt = os.path.join(DATA_DIR, 'patterns.npy')
    if not (os.path.exists(ap) and os.path.exists(pt)):
        raise FileNotFoundError("请先运行 python generate_data.py")

    dataset = DiffractionDataset(ap, pt)
    n = len(dataset)
    n_val = int(n * VAL_RATIO)
    n_train = n - n_val

    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    print(f"训练集 {n_train} / 验证集 {n_val} (seed={SEED}, 与 train.py 一致)")
    return dataset, train_set, val_set


# ============================================================
# 4. 训练去年的 pix2pix
# ============================================================
def train_pix2pix(train_set, val_set):
    G = Pix2PixGenerator().to(DEVICE)
    D = Pix2PixDiscriminator().to(DEVICE)

    opt_g = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)

    hist = {'train_l1': [], 'val_l1': []}

    for epoch in range(EPOCHS):
        G.train(); D.train()
        sum_l1, n_batch = 0.0, 0
        loop = tqdm(train_loader, desc=f"[pix2pix] Epoch {epoch+1}/{EPOCHS}",
                    leave=False)

        for a, p in loop:
            a = to_3ch(a.to(DEVICE))   # 输入: 光阑  3ch [-1,1]
            p = to_3ch(p.to(DEVICE))   # 目标: 衍射  3ch [-1,1]

            # ---- 训练判别器 ----
            opt_d.zero_grad()
            real_pred = D(a, p)
            fake = G(a)
            fake_pred = D(a, fake.detach())
            loss_d = 0.5 * (bce(real_pred, torch.ones_like(real_pred)) +
                            bce(fake_pred, torch.zeros_like(fake_pred)))
            loss_d.backward(); opt_d.step()

            # ---- 训练生成器 ----
            opt_g.zero_grad()
            fake_pred = D(a, fake)
            loss_adv = bce(fake_pred, torch.ones_like(fake_pred))
            loss_l1 = l1(fake, p) * LAMBDA_L1
            loss_g = loss_adv + loss_l1
            loss_g.backward(); opt_g.step()

            sum_l1 += l1(fake.detach(), p).item()
            n_batch += 1
            loop.set_postfix(L1=f"{l1(fake, p).item():.4f}",
                             D=f"{loss_d.item():.4f}")

        train_l1 = sum_l1 / max(n_batch, 1)

        # ---- 验证 ----
        G.eval()
        v_l1, vn = 0.0, 0
        with torch.no_grad():
            for a, p in val_loader:
                a = to_3ch(a.to(DEVICE)); p = to_3ch(p.to(DEVICE))
                fake = G(a)
                v_l1 += l1(fake, p).item() * a.size(0)
                vn += a.size(0)
        val_l1 = v_l1 / vn

        hist['train_l1'].append(train_l1)
        hist['val_l1'].append(val_l1)
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train L1={train_l1:.5f}  val L1={val_l1:.5f}")

    torch.save(G.state_dict(), PIX2PIX_MODEL_PATH)
    print(f"去年 pix2pix 模型已保存: {PIX2PIX_MODEL_PATH}")
    return G, hist


# ============================================================
# 5. 加载今年的物理引导模型
# ============================================================
def load_current():
    if not os.path.exists(CURRENT_MODEL_PATH):
        raise FileNotFoundError(
            f"找不到 {CURRENT_MODEL_PATH}, 请先运行 python train.py")
    model = PhysicsGuidedModel(log_compress=True, alpha=LOG_ALPHA).to(DEVICE)
    model.load_state_dict(torch.load(CURRENT_MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


# ============================================================
# 6. 对比可视化
# ============================================================
@torch.no_grad()
def compare(current_model, pix2pix_G, val_set, n_show=5):
    pix2pix_G.eval()
    idx = np.random.RandomState(0).choice(len(val_set), n_show, replace=False)

    rows = []
    metrics = {
        'phys': {'l1': [], 'mse': []},
        'curr': {'l1': [], 'mse': []},
        'pix2': {'l1': [], 'mse': []},
    }

    for i in idx:
        a, p = val_set[int(i)]
        a_gpu = a.unsqueeze(0).to(DEVICE)

        I_pred, I_sim, delta = current_model(a_gpu)
        I_pred = I_pred.squeeze().cpu().numpy()
        I_sim  = I_sim.squeeze().cpu().numpy()

        fake3 = pix2pix_G(to_3ch(a_gpu))
        I_pix  = from_3ch(fake3).squeeze().cpu().numpy()

        p_np = p.squeeze().numpy()
        a_np = a.squeeze().numpy()

        for key, pred in [('phys', I_sim), ('curr', I_pred), ('pix2', I_pix)]:
            metrics[key]['l1'].append(np.abs(pred - p_np).mean())
            metrics[key]['mse'].append(((pred - p_np) ** 2).mean())

        rows.append((a_np, p_np, I_sim, I_pred, I_pix))

    # ---- 画图: 6 列 ----
    fig, axes = plt.subplots(n_show, 6, figsize=(24, 4 * n_show))
    titles = ['Aperture', 'Ground Truth',
              'Physics Layer', 'Current (phy+res)',
              'Last-year pix2pix', '|Error|  curr vs pix2pix']
    cmaps = ['gray', 'inferno', 'inferno', 'inferno', 'inferno', 'RdBu_r']

    for r, (a_np, p_np, I_sim, I_pred, I_pix) in enumerate(rows):
        err_curr = np.abs(I_pred - p_np)
        err_pix  = np.abs(I_pix  - p_np)
        diff_img = err_curr - err_pix   # 正=今年更差, 负=去年更差

        imgs = [a_np, p_np, I_sim, I_pred, I_pix, diff_img]

        for c in range(6):
            ax = axes[r][c]
            if c == 5:
                v = np.abs(diff_img).max() + 1e-9
                im = ax.imshow(diff_img, cmap='RdBu_r',
                               vmin=-v, vmax=v)
            else:
                im = ax.imshow(imgs[c], cmap=cmaps[c], vmin=0, vmax=1)
            if r == 0:
                ax.set_title(titles[c], fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axes[r][5].set_xlabel(
            f"curr L1={err_curr.mean():.5f} | pix2pix L1={err_pix.mean():.5f}",
            fontsize=9)

    plt.tight_layout()
    plt.savefig(COMPARE_FIG, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存: {COMPARE_FIG}")

    # ---- 打印汇总 ----
    print("\n" + "=" * 70)
    print("验证集前 {} 个样本 定量对比".format(n_show))
    print("=" * 70)
    print(f"{'方案':<22} {'L1 (均值)':>15} {'MSE (均值)':>15}")
    print("-" * 60)
    name_map = {
        'phys': '物理层 (无学习)',
        'curr': '今年: 物理层+残差U-Net',
        'pix2': '去年: pix2pix (U-Net+GAN)',
    }
    for k in ['phys', 'curr', 'pix2']:
        print(f"{name_map[k]:<22} "
              f"{np.mean(metrics[k]['l1']):>15.6f} "
              f"{np.mean(metrics[k]['mse']):>15.6f}")

    # 相对物理层的改善
    base_l1 = np.mean(metrics['phys']['l1'])
    print("\n相对物理层的 L1 改善:")
    for k in ['curr', 'pix2']:
        imp = (base_l1 - np.mean(metrics[k]['l1'])) / base_l1 * 100
        print(f"  {name_map[k]:<30}: {imp:+.1f}%")

    # ---- 画残差对比 ----
    return metrics


# ============================================================
# 7. 训练曲线
# ============================================================
def plot_curves(hist_pix2pix):
    plt.figure(figsize=(10, 5))
    ep = np.arange(1, len(hist_pix2pix['train_l1']) + 1)
    plt.plot(ep, hist_pix2pix['train_l1'], 'o-', label='pix2pix train L1')
    plt.plot(ep, hist_pix2pix['val_l1'],   's-', label='pix2pix val L1')
    plt.yscale('log')
    plt.xlabel('Epoch'); plt.ylabel('L1 (log scale)')
    plt.title('Last-year pix2pix training curve')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURVE_FIG, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"曲线已保存: {CURVE_FIG}")


# ============================================================
# 8. 主流程
# ============================================================
def main():
    print(f"设备: {DEVICE}")
    print(f"LOG_ALPHA = {LOG_ALPHA}\n")

    _, train_set, val_set = load_split()

    # ---- 训练去年 pix2pix (存在则跳过) ----
    if os.path.exists(PIX2PIX_MODEL_PATH):
        print(f"发现已训练 pix2pix, 直接加载: {PIX2PIX_MODEL_PATH}")
        G = Pix2PixGenerator().to(DEVICE)
        G.load_state_dict(torch.load(PIX2PIX_MODEL_PATH, map_location=DEVICE))
        hist = {'train_l1': [0], 'val_l1': [0]}  # 占位
    else:
        print("=" * 60)
        print("训练去年 pix2pix (光阑 -> 衍射)")
        print("=" * 60)
        G, hist = train_pix2pix(train_set, val_set)
        plot_curves(hist)

    # ---- 加载今年模型 ----
    print("\n加载今年物理引导模型...")
    current = load_current()

    # ---- 对比 ----
    print("\n" + "=" * 60)
    print("对比: 物理层 / 今年物理引导 / 去年 pix2pix")
    print("=" * 60)
    compare(current, G, val_set, n_show=5)


if __name__ == '__main__':
    main()