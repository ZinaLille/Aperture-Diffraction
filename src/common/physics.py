"""Physical coordinates and scalar propagation on a common detector grid.

The reference is a scaled angular-spectrum evaluation. The exact propagating
transfer function is factored into a Fresnel transfer and a nonparaxial
correction; the Fresnel part is evaluated by its scaled single-FFT integral.
This avoids enormous periodic FFT windows at small Fresnel numbers. Finite
sampling/window errors remain: validate.py checks padding, refinement and an
independent Rayleigh-Sommerfeld quadrature. This is not a Maxwell solver.
"""
from dataclasses import asdict, dataclass
import numpy as np

DEFAULT_NF = (0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 5.0)


@dataclass(frozen=True)
class PhysicsConfig:
    grid: int = 256
    input_pitch_m: float = 10e-6
    wavelength_m: float = 532e-9
    far_threshold: float = 0.1
    oversample: int = 2
    pad_factor: int = 2
    log_alpha: float = 100.0

    def __post_init__(self):
        if self.grid < 32 or self.grid % 8:
            raise ValueError("grid must be >=32 and divisible by 8")
        if any(not np.isfinite(x) or x <= 0 for x in (self.input_pitch_m, self.wavelength_m, self.log_alpha)):
            raise ValueError("pitch, wavelength and log_alpha must be positive")
        if self.oversample < 1 or self.pad_factor < 1:
            raise ValueError("oversample and pad_factor must be >=1")
        if not 0 < self.far_threshold <= 5:
            raise ValueError("far_threshold must be in (0,5]")

    def to_dict(self):
        return asdict(self)


def check_nf(values):
    values = tuple(float(x) for x in values)
    if not values or any(not np.isfinite(x) or not 0 < x <= 5 for x in values):
        raise ValueError("Fresnel numbers must be finite and in (0,5.0]")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate Fresnel numbers")
    if len(set(format(x, ".6g") for x in values)) != len(values):
        raise ValueError("Fresnel numbers are too close for distinct group directory names")
    return values


def group_name(nf):
    return "nf_" + format(float(nf), ".6g").replace(".", "p")


def support_width(aperture, pitch):
    """D = largest axis-aligned occupied pixel span, including inter-hole gaps."""
    rows, cols = np.nonzero(aperture > 0)
    if not len(rows):
        raise ValueError("An empty aperture has no defined Fresnel number")
    return max(np.ptp(rows) + 1, np.ptp(cols) + 1) * pitch


def geometry(aperture, nf, config):
    check_nf([nf])
    d = support_width(aperture, config.input_pitch_m)
    z = d * d / (config.wavelength_m * nf)
    detector_pitch = config.wavelength_m * z / (config.grid * config.input_pitch_m)
    return {"D_m": d, "z_m": z, "nf": float(nf),
            "detector_pitch_m": detector_pitch,
            "detector_fov_m": config.grid * detector_pitch}


def postprocess(raw, alpha=100.0):
    raw = np.maximum(np.asarray(raw, dtype=np.float64), 0)
    peak = raw.max()
    if not np.isfinite(peak):
        raise ValueError("Nonfinite propagated intensity")
    if peak == 0:
        return np.zeros_like(raw, dtype=np.float32)
    value = np.log1p(alpha * (raw / peak))
    return (value / value.max()).astype(np.float32)


def fraunhofer_raw(aperture, z, config):
    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(aperture)))
    return np.abs(field * config.input_pitch_m**2 / (config.wavelength_m * z))**2


def _fine_aperture(aperture, oversample, pad_factor):
    # Aperture pixels represent piecewise-constant cells. Midpoint quadrature
    # needs a half-fine-pixel origin correction when oversample is even.
    fine = np.repeat(np.repeat(aperture, oversample, 0), oversample, 1)
    m = len(fine) * pad_factor
    result = np.zeros((m, m), dtype=np.float64)
    offset = (m - len(fine)) // 2
    result[offset:offset+len(fine), offset:offset+len(fine)] = fine
    origin_shift = (oversample - 1) / 2
    return result, origin_shift


def angular_spectrum_raw(aperture, z, config):
    """Exact angular dispersion, evaluated on the Fraunhofer detector grid."""
    c, pad = config.oversample, config.pad_factor
    fine, origin_shift = _fine_aperture(aperture, c, pad)
    m = len(fine)
    dx = config.input_pitch_m / c
    frequency = np.fft.fftfreq(m, dx)
    q = config.wavelength_m**2 * (frequency[:, None]**2 + frequency[None, :]**2)
    propagating = q < 1
    root = np.sqrt(np.maximum(1 - q, 0))
    # Stable evaluation of sqrt(1-q)-1+q/2 (avoids cancellation).
    residual_phase = -(2*np.pi/config.wavelength_m)*z*q*q/(2*(1+root)**2)
    correction = np.exp(1j * residual_phase) * propagating
    spectrum = np.fft.fft2(np.fft.ifftshift(fine))
    corrected = np.fft.fftshift(np.fft.ifft2(spectrum * correction))
    coords = (np.arange(m) - m//2 - origin_shift) * dx
    chirp_1d = np.exp(1j*np.pi*coords**2/(config.wavelength_m*z))
    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
        corrected * chirp_1d[:, None] * chirp_1d[None, :])))
    # Output x_j=(j-N/2)*lambda*z/(N*input_pitch). Zero-padding samples this
    # same grid more densely; selecting every pad-th point keeps it unchanged.
    indices = m//2 + (np.arange(config.grid)-config.grid//2)*pad
    field = field[np.ix_(indices, indices)]
    # The subcell-origin shift changes output phase only, not its intensity.
    return np.abs(field * dx**2/(config.wavelength_m*z))**2


def rayleigh_sommerfeld_points(aperture, z, points_xy, config, oversample=4):
    """Independent scalar RS1 midpoint quadrature; small validation cases only.

    Kernel includes the 1/(kr) term. Global exp(ikz) is omitted. points_xy
    contains physical detector positions, not array indices.
    """
    fine = np.repeat(np.repeat(aperture, oversample, 0), oversample, 1)
    dx = config.input_pitch_m / oversample
    origin = (oversample-1)/2
    coords = (np.arange(len(fine))-len(fine)//2-origin)*dx
    iy, ix = np.nonzero(fine)
    xx, yy, weights = coords[ix], coords[iy], fine[iy, ix]
    k = 2*np.pi/config.wavelength_m
    values = []
    for x, y in points_xy:
        rho2 = (x-xx)**2 + (y-yy)**2
        r = np.sqrt(z*z+rho2)
        # r-z=rho²/(r+z), stable at long distance.
        phase = k*rho2/(r+z)
        kernel = z/(1j*config.wavelength_m*r*r)*(1+1j/(k*r))*np.exp(1j*phase)
        values.append(abs(np.sum(weights*kernel)*dx*dx)**2)
    return np.asarray(values)


def image_metrics(pred, target):
    difference = np.asarray(pred, dtype=np.float64)-np.asarray(target, dtype=np.float64)
    mae = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference**2)))
    # Fixed range, unlike legacy PSNR's prediction-dependent peak.
    psnr = float(20*np.log10(1/max(rmse, 1e-12)))
    corr = float(np.corrcoef(np.ravel(pred), np.ravel(target))[0, 1]) if min(np.std(pred), np.std(target)) > 1e-12 else 0.0
    return {"mae": mae, "rmse": rmse, "psnr_db": psnr, "corr": corr}
