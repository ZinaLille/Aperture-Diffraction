"""Differentiable scaled ASM matching physics.angular_spectrum_raw coordinates.

No trainable parameters. All deterministic buffers are rebuilt from config and
omitted from checkpoints. z is an actual distance in metres, NEVER a normalized
network condition. A residual added afterwards is still unconstrained.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .physics import PhysicsConfig


class AngularSpectrumDetectorLayer(nn.Module):
    def __init__(self, config=None, precision="float32"):
        super().__init__()
        self.config = config or PhysicsConfig()
        if precision not in ("float32", "float64"):
            raise ValueError("ASM precision must be float32 or float64")
        self.precision = precision
        self.real_dtype = torch.float64 if precision == "float64" else torch.float32
        self.complex_dtype = torch.complex128 if precision == "float64" else torch.complex64
        self.alpha = self.config.log_alpha
        c, pad, n = self.config.oversample, self.config.pad_factor, self.config.grid
        self.fine_grid = n*c
        self.padded_grid = m = self.fine_grid*pad
        dx = self.config.input_pitch_m/c
        frequency = torch.fft.fftfreq(m, d=dx, dtype=torch.float64)
        q = self.config.wavelength_m**2*(frequency[:, None]**2+frequency[None, :]**2)
        root = torch.sqrt(torch.clamp(1-q, min=0))
        coefficient = -(2*torch.pi/self.config.wavelength_m)*q.square()/(2*(1+root).square())
        coord = (torch.arange(m, dtype=torch.float64)-m//2-(c-1)/2)*dx
        self.register_buffer("correction_coefficient", coefficient, persistent=False)
        self.register_buffer("propagating", q < 1, persistent=False)
        self.register_buffer("chirp_coefficient", torch.pi*coord.square()/self.config.wavelength_m, persistent=False)
        self.register_buffer("output_indices", m//2+(torch.arange(n)-n//2)*pad, persistent=False)

    def raw_intensity(self, aperture, z_m):
        if aperture.ndim != 4 or aperture.shape[1:] != (1, self.config.grid, self.config.grid):
            raise ValueError("ASM expects [B,1,grid,grid] matching physics config")
        if z_m is None:
            raise ValueError("ASM requires physical z_m in metres")
        z = torch.as_tensor(z_m, dtype=torch.float64, device=aperture.device).reshape(-1)
        if z.numel() != len(aperture) or not torch.isfinite(z).all() or (z <= 0).any():
            raise ValueError("Need one finite positive propagation distance per sample")
        c = self.config.oversample
        fine = aperture[:, 0].to(self.real_dtype).repeat_interleave(c, -2).repeat_interleave(c, -1)
        offset = (self.padded_grid-self.fine_grid)//2
        fine = F.pad(fine, (offset, offset, offset, offset))
        spectrum = torch.fft.fft2(torch.fft.ifftshift(fine, dim=(-2, -1)))
        phase = z[:, None, None]*self.correction_coefficient
        correction = torch.polar(torch.ones_like(phase), phase).to(self.complex_dtype)
        correction = correction*self.propagating
        corrected = torch.fft.fftshift(torch.fft.ifft2(spectrum*correction), dim=(-2, -1))
        phase = self.chirp_coefficient[None, :]/z[:, None]
        chirp = torch.polar(torch.ones_like(phase), phase).to(self.complex_dtype)
        field = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(
            corrected*chirp[:, :, None]*chirp[:, None, :], dim=(-2, -1))), dim=(-2, -1))
        field = field.index_select(-2, self.output_indices).index_select(-1, self.output_indices)
        scale = (self.config.input_pitch_m/c)**2/(self.config.wavelength_m*z)
        return (field.abs().square()*scale[:, None, None].to(self.real_dtype).square()).unsqueeze(1)

    def forward(self, aperture, z_m):
        raw = self.raw_intensity(aperture, z_m)
        value = raw/raw.amax((-2, -1), keepdim=True).clamp_min(1e-30)
        value = torch.log1p(self.alpha*value)
        value = value/value.amax((-2, -1), keepdim=True).clamp_min(1e-30)
        # Network weights remain float32 even when reference propagation uses fp64.
        return value.to(aperture.dtype)
