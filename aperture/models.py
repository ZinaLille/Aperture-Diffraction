"""U-Net blocks, stage-one forward models and the image loss."""
import torch
from torch import nn

from .physics import PhysicsConfig, AngularSpectrumDetectorLayer

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


class ConditionedUNet(nn.Module):
    def __init__(self, in_channels, base_ch=32, residual=False, target_mean=0.0):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)
        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, 2)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2)
        self.dec3 = ConvBlock(base_ch*8, base_ch*4)
        self.dec2 = ConvBlock(base_ch*4, base_ch*2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 1)
        if residual:
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)
        else:
            nn.init.normal_(self.out_conv.weight, 0, 1e-3)
            nn.init.constant_(self.out_conv.bias, target_mean)

    def forward(self, value):
        e1 = self.enc1(value)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        value = self.bottleneck(self.pool(e3))
        value = self.dec3(torch.cat((self.up3(value), e3), 1))
        value = self.dec2(torch.cat((self.up2(value), e2), 1))
        value = self.dec1(torch.cat((self.up1(value), e1), 1))
        return self.out_conv(value)


class FraunhoferDetectorLayer(nn.Module):
    """Normalized output on x=lambda*z*f, not an unregistered camera plane.

    Detector pitch is chosen as lambda*z/(N*input_pitch), so its array values
    are the normalized aperture FFT for every sample. Absolute propagation
    factors cancel in this representation's per-image max normalization.
    """
    def __init__(self, alpha=100.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, aperture, z_m=None):
        field = torch.fft.fft2(aperture.squeeze(1))
        intensity = torch.fft.fftshift(field, dim=(-2, -1)).abs().square()
        intensity = intensity/intensity.amax((-2, -1), keepdim=True).clamp_min(1e-30)
        intensity = torch.log1p(self.alpha*intensity)
        intensity = intensity/intensity.amax((-2, -1), keepdim=True).clamp_min(1e-30)
        return intensity.unsqueeze(1)


class ForwardModel(nn.Module):
    def __init__(self, kind, base_ch=32, alpha=100.0, target_mean=0.0,
                 propagator="fraunhofer", physics_config=None, asm_precision="float32"):
        super().__init__()
        if kind not in ("physics", "baseline"):
            raise ValueError(kind)
        self.kind = kind
        self.propagator = propagator
        if propagator not in ("fraunhofer", "asm"):
            raise ValueError(propagator)
        self.physics = (AngularSpectrumDetectorLayer(physics_config or PhysicsConfig(log_alpha=alpha), asm_precision)
                        if propagator == "asm" and kind == "physics" else FraunhoferDetectorLayer(alpha))
        self.network = ConditionedUNet(4 if kind == "physics" else 3, base_ch,
                                       residual=kind == "physics", target_mean=target_mean)

    def forward(self, aperture, condition, z_m=None):
        # condition contains train-normalized log(z) and log(detector pitch).
        maps = condition[:, :, None, None].expand(-1, -1, *aperture.shape[-2:])
        if self.kind == "physics":
            simulated = self.physics(aperture, z_m)
            delta = self.network(torch.cat((aperture, simulated, maps), 1))
            return simulated+delta, simulated, delta
        predicted = self.network(torch.cat((aperture, maps), 1))
        return predicted, None, None
