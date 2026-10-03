"""
train.py (修正版)

配套 generate_data_v3.py 使用的训练脚本。

关键修正:
    物理层 FraunhoferLayer 加了 log 压缩，
    和 generate_data_v3.simulate_diffraction 完全一致。
    
    修正前: 物理层输出线性强度 → 残差网络被迫学 log 变换
    修正后: 物理层直接输出 log 压缩强度 → 残差网络接近零输出

运行:
    python train.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ============================================================
# 配置
# ============================================================
DATA_DIR = 'data'
SAVE_PATH = 'best_model.pt'
RESULT_PATH = 'train_result.png'

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
VAL_RATIO = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ⚠️ 必须和 generate_data_v3.py 里的 LOG_ALPHA 一致
LOG_ALPHA = 100.0


# ============================================================
# 1. 数据集
# ============================================================
class DiffractionDataset(Dataset):
    """
    从 .npy 文件加载配对数据。
    
    数据格式 (由 generate_data_v3.py 生成):
        apertures.npy: (N, 256, 256) float32, 值域 [0, 1]
        patterns.npy:  (N, 256, 256) float32, 值域 [0, 1]  (已 log 压缩)
    """
    def __init__(self, aperture_path, pattern_path):
        aps = np.load(aperture_path)
        pats = np.load(pattern_path)
        
        assert len(aps) == len(pats), "光阑和衍射图样数量不一致"
        
        print(f"加载数据: {len(aps)} 组")
        print(f"  光阑:  {aps.shape}, 范围 [{aps.min():.3f}, {aps.max():.3f}]")
        print(f"  衍射:  {pats.shape}, 范围 [{pats.min():.3f}, {pats.max():.3f}]")
        
        self.apertures = torch.tensor(aps, dtype=torch.float32).unsqueeze(1)
        self.patterns = torch.tensor(pats, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.apertures)
    
    def __getitem__(self, idx):
        return self.apertures[idx], self.patterns[idx]


# ============================================================
# 2. 物理层: 夫琅禾费衍射 (FFT + log 压缩)
# ============================================================
class FraunhoferLayer(nn.Module):
    """
    夫琅禾费衍射物理层。
    
    数学: I(u,v) = |FFT{a(x,y)}|²
    
    处理流程 (必须和 generate_data_v3.simulate_diffraction 完全一致):
        1. FFT + fftshift
        2. |.|²
        3. 归一化到 [0, 1] (除以 max)
        4. log 压缩: log(1 + α·I)
        5. 再归一化到 [0, 1]
    
    无参数，完全可微。
    """
    def __init__(self, log_compress=True, alpha=LOG_ALPHA):
        super().__init__()
        self.log_compress = log_compress
        self.alpha = alpha
    
    def forward(self, aperture):
        """
        Args:
            aperture: (B, 1, H, W) 实数张量
        Returns:
            intensity: (B, 1, H, W) 归一化后的 log 压缩强度
        """
        # 转复数
        field = aperture.squeeze(1).to(torch.complex64)   # (B, H, W)
        
        # FFT + fftshift
        field_f = torch.fft.fft2(field)
        field_f = torch.fft.fftshift(field_f, dim=(-2, -1))
        
        # 强度
        intensity = torch.abs(field_f) ** 2               # (B, H, W)
        
        # === 第 1 次归一化 ===
        max_val = intensity.amax(dim=(-2, -1), keepdim=True) + 1e-12
        intensity = intensity / max_val
        
        # === log 压缩 (修正点！) ===
        if self.log_compress:
            intensity = torch.log1p(self.alpha * intensity)
            max_val = intensity.amax(dim=(-2, -1), keepdim=True) + 1e-12
            intensity = intensity / max_val
        
        return intensity.unsqueeze(1)                     # (B, 1, H, W)


# ============================================================
# 3. 残差网络 (轻量 U-Net)
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


class ResidualUNet(nn.Module):
    """
    输入: 光阑 a + 模拟图 I_sim, 拼接后 2 通道
    输出: 残差 ΔI
    
    关键: 输出层初始化为 0，
          训练开始时 I_pred = I_sim + 0 = I_sim
    """
    def __init__(self, base_ch=32):
        super().__init__()
        
        self.enc1 = ConvBlock(2, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)
        
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        
        self.out_conv = nn.Conv2d(base_ch, 1, 1)
        
        # 输出初始化为 0，训练从纯物理开始
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, a, I_sim):
        x = torch.cat([a, I_sim], dim=1)          # (B, 2, H, W)
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        
        d3 = self.up3(b);  d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3); d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.out_conv(d1)                  # (B, 1, H, W)


# ============================================================
# 4. 完整模型
# ============================================================
class PhysicsGuidedModel(nn.Module):
    """
    光阑 -> FFT物理层 -> I_sim
         -> 残差网络 -> ΔI
         -> I_pred = I_sim + ΔI
    """
    def __init__(self, log_compress=True, alpha=LOG_ALPHA):
        super().__init__()
        self.physics = FraunhoferLayer(log_compress=log_compress, alpha=alpha)
        self.residual = ResidualUNet(base_ch=32)
    
    def forward(self, a):
        I_sim = self.physics(a)
        delta = self.residual(a, I_sim)
        I_pred = I_sim + delta
        return I_pred, I_sim, delta
    
    def freeze_physics(self):
        """物理层无参数，此接口保留以保持兼容"""
        for param in self.physics.parameters():
            param.requires_grad = False

# ============================================================
# 5. 损失函数 (改进版)
# ============================================================
class Loss(nn.Module):
    """
    L = L1(I_pred, I_tgt)
      + λ_f · L1(log|FFT(I_pred)|, log|FFT(I_tgt)|)
      + λ_s · (1 - SSIM)
      + λ_r · |ΔI|          (仅当传入 delta 时生效)

    改进点 (相比原始版):
        1. 频域取 log 幅度 → 旁瓣梯度不再被中心峰掩盖
        2. 加 SSIM → 约束几何结构 (Airy 环/光栅级次)
        3. 加残差 L1 正则 → 鼓励网络输出接近 0,
           防止网络"重学"物理层已经算好的东西
        4. 各项可单独开关, 便于做 ablation

    用法:
        criterion = Loss(lambda_freq=0.1, lambda_ssim=0.2, lambda_res=1e-3)
        loss, stats = criterion(I_pred, I_target, delta=delta)
    """

    def __init__(self,
                 lambda_freq=0.1,
                 lambda_ssim=0.2,
                 lambda_res=1e-3,
                 use_log_freq=True):
        super().__init__()
        self.lambda_freq = lambda_freq
        self.lambda_ssim = lambda_ssim
        self.lambda_res  = lambda_res
        self.use_log_freq = use_log_freq
        self.l1 = nn.L1Loss()

        # SSIM (延迟导入, 没装也不影响前两项)
        self._ssim_fn = None
        if self.lambda_ssim > 0:
            try:
                from torchmetrics.functional import structural_similarity as ssim_fn
                self._ssim_fn = ssim_fn
            except ImportError:
                try:
                    from pytorch_msssim import ssim as ssim_fn
                    self._ssim_fn = lambda x, y: ssim_fn(
                        x, y, data_range=1.0, size_average=True)
                except ImportError:
                    print("⚠ 未安装 SSIM 依赖, lambda_ssim 自动置 0")
                    print("  pip install torchmetrics  (推荐)")
                    print("  或 pip install pytorch-msssim")
                    self.lambda_ssim = 0.0

    def forward(self, I_pred, I_target, delta=None):
        # ---------- 1) 空间 L1 ----------
        loss_data = self.l1(I_pred, I_target)

        # ---------- 2) 频域 log 幅度 L1 ----------
        if self.lambda_freq > 0:
            F_pred = torch.fft.fft2(I_pred.squeeze(1))
            F_target = torch.fft.fft2(I_target.squeeze(1))
            A_pred = torch.abs(F_pred)
            A_target = torch.abs(F_target)
            if self.use_log_freq:
                A_pred = torch.log1p(A_pred)
                A_target = torch.log1p(A_target)
            loss_freq = torch.mean(torch.abs(A_pred - A_target))
        else:
            loss_freq = torch.tensor(0.0, device=I_pred.device)

        # ---------- 3) SSIM ----------
        if self.lambda_ssim > 0 and self._ssim_fn is not None:
            ssim_val = self._ssim_fn(I_pred, I_target)
            loss_ssim = 1.0 - ssim_val
        else:
            loss_ssim = torch.tensor(0.0, device=I_pred.device)

        # ---------- 4) 残差正则 ----------
        if self.lambda_res > 0 and delta is not None:
            loss_res = torch.mean(torch.abs(delta))
        else:
            loss_res = torch.tensor(0.0, device=I_pred.device)

        total = (loss_data
                 + self.lambda_freq * loss_freq
                 + self.lambda_ssim * loss_ssim
                 + self.lambda_res  * loss_res)

        stats = {
            'l1':   loss_data.item(),
            'freq': loss_freq.item(),
            'ssim': loss_ssim.item(),
            'res':  loss_res.item(),
        }
        return total, stats

# ============================================================
# 6. 训练与验证
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    agg = {'l1': 0.0, 'freq': 0.0, 'ssim': 0.0, 'res': 0.0}
    n = 0
    for a, p in loader:
        a, p = a.to(device), p.to(device)

        optimizer.zero_grad()
        I_pred, _, delta = model(a)
        loss, stats = criterion(I_pred, p, delta=delta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k in agg:
            agg[k] += stats[k] * a.size(0)
        n += a.size(0)

    return {k: v / n for k, v in agg.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    agg = {'l1': 0.0, 'freq': 0.0, 'ssim': 0.0, 'res': 0.0}
    n = 0
    for a, p in loader:
        a, p = a.to(device), p.to(device)
        I_pred, _, delta = model(a)
        _, stats = criterion(I_pred, p, delta=delta)
        for k in agg:
            agg[k] += stats[k] * a.size(0)
        n += a.size(0)
    return {k: v / n for k, v in agg.items()}

# ============================================================
# 7. 可视化
# ============================================================
def visualize(model, dataset, device, save_path='train_result.png', n_show=4):
    """
    5 列可视化:
        光阑 | 物理层 | 预测 | 真值 | 残差
    
    修正后预期:
        第 2 列 (物理层) 和 第 4 列 (真值) 几乎一样
        第 3 列 (预测) 和 第 2 列 几乎一样
        第 5 列 (残差) 接近 0
    """
    model.eval()
    fig, axes = plt.subplots(n_show, 5, figsize=(20, 4 * n_show))
    
    indices = np.random.choice(len(dataset), n_show, replace=False)
    
    for row, idx in enumerate(indices):
        a, p = dataset[idx]
        a_in = a.unsqueeze(0).to(device)
        p_np = p.squeeze().numpy()
        
        with torch.no_grad():
            I_pred, I_sim, delta = model(a_in)
        I_pred = I_pred.squeeze().cpu().numpy()
        I_sim = I_sim.squeeze().cpu().numpy()
        delta = delta.squeeze().cpu().numpy()
        a_np = a.squeeze().numpy()
        
        titles = ['Aperture', 'I_sim (physics)', 'I_pred (physics+res)',
                  'Ground truth', 'Residual ΔI']
        imgs = [a_np, I_sim, I_pred, p_np, delta]
        cmaps = ['gray', 'inferno', 'inferno', 'inferno', 'RdBu_r']
        
        for col in range(5):
            ax = axes[row][col]
            if cmaps[col] == 'RdBu_r':
                vmax = np.abs(delta).max() + 1e-9
                im = ax.imshow(delta, cmap=cmaps[col], vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(imgs[col], cmap=cmaps[col], vmin=0, vmax=1)
            if row == 0:
                ax.set_title(titles[col], fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 在残差图上加文字
        sim_mae = np.mean(np.abs(I_sim - p_np))
        pred_mae = np.mean(np.abs(I_pred - p_np))
        axes[row][4].set_xlabel(
            f"sim MAE={sim_mae:.5f}  pred MAE={pred_mae:.5f}",
            fontsize=9
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"可视化保存到: {save_path}")


def plot_loss_curve(history, save_path='loss_curve.png'):
    """
    多子图 loss 曲线。
    history 为 dict, 键形如 'train_l1' / 'val_l1' 等。
    """
    has_freq = 'train_freq' in history
    has_ssim = 'train_ssim' in history
    has_res  = 'train_res'  in history
    n_panels = 1 + int(has_freq) + int(has_ssim) + int(has_res)
    
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    
    # panel 1: L1
    ax = axes[0]
    ax.plot(history['train_l1'], label='train', lw=1.5)
    ax.plot(history['val_l1'],   label='val',   lw=1.5)
    ax.set_xlabel('epoch'); ax.set_ylabel('L1')
    ax.set_yscale('log')
    ax.set_title('L1 loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    idx = 1
    if has_freq:
        ax = axes[idx]; idx += 1
        ax.plot(history['train_freq'], label='train', lw=1.5)
        ax.plot(history['val_freq'],   label='val',   lw=1.5)
        ax.set_xlabel('epoch'); ax.set_ylabel('freq')
        ax.set_yscale('log')
        ax.set_title('Freq log-magnitude L1')
        ax.legend(); ax.grid(True, alpha=0.3)
    
    if has_ssim:
        ax = axes[idx]; idx += 1
        ax.plot(history['train_ssim'], label='train', lw=1.5)
        ax.plot(history['val_ssim'],   label='val',   lw=1.5)
        ax.set_xlabel('epoch'); ax.set_ylabel('1 - SSIM')
        ax.set_yscale('log')
        ax.set_title('SSIM loss')
        ax.legend(); ax.grid(True, alpha=0.3)
    
    if has_res:
        ax = axes[idx]; idx += 1
        ax.plot(history['train_res'], label='train', lw=1.5)
        ax.plot(history['val_res'],   label='val',   lw=1.5)
        ax.set_xlabel('epoch'); ax.set_ylabel('|ΔI|')
        ax.set_yscale('log')
        ax.set_title('Residual regularization')
        ax.legend(); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Loss 曲线保存到: {save_path}")

# ============================================================
# 8. 主流程
# ============================================================
def main():
    print(f"设备: {DEVICE}")
    print(f"LOG_ALPHA = {LOG_ALPHA}  (必须和 generate_data_v3.py 一致)")
    print()
    
    # ---------- 加载数据 ----------
    ap_path = os.path.join(DATA_DIR, 'apertures.npy')
    pat_path = os.path.join(DATA_DIR, 'patterns.npy')
    
    if not os.path.exists(ap_path) or not os.path.exists(pat_path):
        raise FileNotFoundError(
            f"找不到 {ap_path} 或 {pat_path}\n"
            f"请先运行: python generate_data_v3.py"
        )
    
    dataset = DiffractionDataset(ap_path, pat_path)
    
    n = len(dataset)
    n_val = int(n * VAL_RATIO)
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"训练集: {n_train}, 验证集: {n_val}")
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=2)
    
    # ---------- 创建模型 ----------
    model = PhysicsGuidedModel(log_compress=True, alpha=LOG_ALPHA).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    residual_params = sum(p.numel() for p in model.residual.parameters())
    print(f"总参数: {total_params:,}  (残差网络: {residual_params:,})")
    
    # ---------- 预检查: 物理层输出是否和训练目标匹配 ----------
    print("\n" + "=" * 60)
    print("预检查: 物理层输出 vs 训练目标 (训练前)")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        a, p = dataset[0]
        a_t = a.unsqueeze(0).to(DEVICE)
        _, I_sim, _ = model(a_t)
        I_sim = I_sim.squeeze().cpu().numpy()
        p_np = p.squeeze().numpy()
        
        mae = np.mean(np.abs(I_sim - p_np))
        corr = np.corrcoef(I_sim.flatten(), p_np.flatten())[0, 1]
        
        print(f"物理层: [{I_sim.min():.4f}, {I_sim.max():.4f}]  mean={I_sim.mean():.4f}")
        print(f"真值:   [{p_np.min():.4f}, {p_np.max():.4f}]  mean={p_np.mean():.4f}")
        print(f"MAE:   {mae:.6f}")
        print(f"Corr:  {corr:.6f}")
        
        if mae < 0.001 and corr > 0.999:
            print("✓ 物理层和训练目标匹配")
        else:
            print("⚠ 物理层和训练目标不匹配！")
            print("  检查:")
            print("    1. LOG_ALPHA 是否和 generate_data_v3.py 一致")
            print("    2. fftshift 顺序是否一致")
            print("    3. 归一化方式是否一致")
            print("  如果预检查不通过，训练下去也不会得到正确结果。")
            return
    
    # ---------- 优化器和损失 ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # ⬇️ 改 1: 使用新的损失配置
    #    λ_f=0.1 : 频域 log 幅度
    #    λ_s=0.2 : SSIM (约束几何结构)
    #    λ_r=1e-3: 残差 L1 正则 (鼓励残差接近 0)
    #    做 ablation 时, 改这里即可 (例如置 0 关闭某项)
    criterion = Loss(
        lambda_freq=0.1,
        lambda_ssim=0.2,
        lambda_res=1e-3,
        use_log_freq=True,
    )
    print(f"\n损失配置: λ_freq={criterion.lambda_freq}, "
          f"λ_ssim={criterion.lambda_ssim}, "
          f"λ_res={criterion.lambda_res}, "
          f"log_freq={criterion.use_log_freq}")
    
    # ---------- 训练 ----------
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    best_val = float('inf')
    # ⬇️ 改 2: history 记录各项损失, 便于后续画多条曲线
    history = {
        'train_l1':   [], 'val_l1':   [],
        'train_freq': [], 'val_freq': [],
        'train_ssim': [], 'val_ssim': [],
        'train_res':  [], 'val_res':  [],
    }
    
    for epoch in range(EPOCHS):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        va = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        # ⬇️ 改 3: 现在 tr / va 是 dict
        history['train_l1'].append(tr['l1'])
        history['val_l1'].append(va['l1'])
        history['train_freq'].append(tr['freq'])
        history['val_freq'].append(va['freq'])
        history['train_ssim'].append(tr['ssim'])
        history['val_ssim'].append(va['ssim'])
        history['train_res'].append(tr['res'])
        history['val_res'].append(va['res'])
        
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train L1: {tr['l1']:.6e} (freq {tr['freq']:.3e}, "
              f"ssim {tr['ssim']:.3e}, res {tr['res']:.3e}) | "
              f"Val L1: {va['l1']:.6e}")
        
        # ⬇️ 改 4: 用 va['l1'] 作为早停/选优指标 (和旧版对齐)
        if va['l1'] < best_val:
            best_val = va['l1']
            torch.save(model.state_dict(), SAVE_PATH)
    
    print(f"\n训练完成，最佳验证 L1: {best_val:.6f}")
    print(f"模型保存到: {SAVE_PATH}")
    
    # ---------- Loss 曲线 ----------
    # ⬇️ 改 5: 传入新的 history 字典 (见下方新 plot_loss_curve)
    plot_loss_curve(history, save_path='loss_curve.png')
    
    # ---------- 加载最佳模型并可视化 ----------
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    visualize(model, val_set, DEVICE, save_path=RESULT_PATH, n_show=4)
    
    # ---------- 打印定量对比 ----------
    print("\n" + "=" * 60)
    print("定量对比 (验证集前 8 个样本)")
    print("=" * 60)
    print(f"{'样本':>4} | {'物理层L1':>10} | {'预测L1':>10} | {'改善':>8}")
    print("-" * 42)
    
    model.eval()
    with torch.no_grad():
        for i in range(min(8, len(val_set))):
            a, p = val_set[i]
            a_in = a.unsqueeze(0).to(DEVICE)
            I_pred, I_sim, _ = model(a_in)
            p_np = p.squeeze().numpy()
            sim_l1 = np.mean(np.abs(I_sim.squeeze().cpu().numpy() - p_np))
            pred_l1 = np.mean(np.abs(I_pred.squeeze().cpu().numpy() - p_np))
            impr = (sim_l1 - pred_l1) / (sim_l1 + 1e-12) * 100
            print(f"  {i:>2} | {sim_l1:>10.6f} | {pred_l1:>10.6f} | {impr:>+7.1f}%")
    
    # ---------- 残差网络输出统计 ----------
    print("\n残差网络输出统计 (验证集前 100 个):")
    residual_magnitudes = []
    with torch.no_grad():
        for i in range(min(100, len(val_set))):
            a, _ = val_set[i]
            a_in = a.unsqueeze(0).to(DEVICE)
            _, _, delta = model(a_in)
            residual_magnitudes.append(
                np.abs(delta.squeeze().cpu().numpy()).mean()
            )
    rm = np.array(residual_magnitudes)
    print(f"  残差绝对值均值: {rm.mean():.6f} ± {rm.std():.6f}")
    print(f"  残差绝对值最大: {rm.max():.6f}")
    
    if rm.mean() < 0.01:
        print("  ✓ 残差接近 0，物理层已足够好")
    else:
        print("  ⚠ 残差偏大，可能物理层还有不匹配")


if __name__ == '__main__':
    main()