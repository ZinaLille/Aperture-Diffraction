"""Neural-network building blocks shared by every phase.

`ConvBlock` is the U-Net building block used by the stage-one residual model,
its pure-network baseline, and the future inverse model. `Loss` is the stage-one
image loss (spatial L1 + log-frequency L1 + optional SSIM + residual L1); the
inverse phase may reuse it for re-propagated pattern consistency.
"""

import torch
from torch import nn


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


class Loss(nn.Module):
    """L = L1(pred, target) + λ_f·L1(log|FFT|) + λ_s·(1-SSIM) + λ_r·|ΔI|.

    All terms can be switched off independently for ablations. SSIM needs
    ``torchmetrics`` or ``pytorch-msssim``; without either one λ_s falls back to
    zero with a warning instead of failing.
    """

    def __init__(self,
                 lambda_freq=0.1,
                 lambda_ssim=0.2,
                 lambda_res=1e-3,
                 use_log_freq=True):
        super().__init__()
        self.lambda_freq = lambda_freq
        self.lambda_ssim = lambda_ssim
        self.lambda_res = lambda_res
        self.use_log_freq = use_log_freq
        self.l1 = nn.L1Loss()

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
        loss_data = self.l1(I_pred, I_target)

        if self.lambda_freq > 0:
            A_pred = torch.abs(torch.fft.fft2(I_pred.squeeze(1)))
            A_target = torch.abs(torch.fft.fft2(I_target.squeeze(1)))
            if self.use_log_freq:
                A_pred = torch.log1p(A_pred)
                A_target = torch.log1p(A_target)
            loss_freq = torch.mean(torch.abs(A_pred - A_target))
        else:
            loss_freq = torch.tensor(0.0, device=I_pred.device)

        if self.lambda_ssim > 0 and self._ssim_fn is not None:
            loss_ssim = 1.0 - self._ssim_fn(I_pred, I_target)
        else:
            loss_ssim = torch.tensor(0.0, device=I_pred.device)

        if self.lambda_res > 0 and delta is not None:
            loss_res = torch.mean(torch.abs(delta))
        else:
            loss_res = torch.tensor(0.0, device=I_pred.device)

        total = (loss_data
                 + self.lambda_freq * loss_freq
                 + self.lambda_ssim * loss_ssim
                 + self.lambda_res * loss_res)

        stats = {
            'l1': loss_data.item(),
            'freq': loss_freq.item(),
            'ssim': loss_ssim.item(),
            'res': loss_res.item(),
        }
        return total, stats
