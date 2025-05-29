from __future__ import annotations
import numpy as np


try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    fft = pyfftw.interfaces.numpy_fft.fft
except ModuleNotFoundError:
    from numpy.fft import fft
from numpy.fft import fftshift, fftfreq


def gaussian(x, w, x0=0.0):
    """L²-normalised Gaussian with 1/e beam radius w."""
    g = np.exp(-2 * ((x - x0) / w) ** 2)
    g /= np.sqrt((np.abs(g)**2).sum())
    return g.astype(np.complex128)


def lens_phase(x, wl, f) -> np.ndarray:
    """Thin-lens quadratic phase  exp(-iπ x² / (λ f))."""
    return np.exp(-1j * np.pi * x ** 2 / (wl * f))


def blazed_mask(x, theta, wl_ref) -> np.ndarray:
    d = wl_ref / np.sin(theta)  # grating period, d*sin(theta)=lambda*m; m=1
    phi = (2 * np.pi / d * x) % (2 * np.pi)
    return np.exp(1j * phi)


def farfield(E, wl, f, dx):
    N = len(E)
    Ef = fftshift(fft(fftshift(E))) * dx          # include sampling
    I  = np.abs(Ef) ** 2

    kx = fftshift(fftfreq(N, d=dx) * 2 * np.pi)
    x_det = f * wl * kx / (2 * np.pi)

    # energy normalisation
    I /= I.sum()
    return x_det, I


def regrid(I_src, x_src, x_ref):
    """Interpolate I(x_src) → I(x_ref), zero outside range."""
    return np.interp(x_ref, x_src, I_src, left=0.0, right=0.0)


# ---------------------------------------------------------------------
class GratingSim1D:
    """
    f        : focal length of the lens [m]
    """

    def __init__(self,
                 Nx: int = 2**14,
                 Lx: float = 4e-3,
                 wl0: float = 808e-9,
                 Dwl: float = 100e-9,
                 N_wl: int = 31,
                 waist: float = 1500e-6,
                 x0: float = 0.0,
                 blaze_angle: float = 0.15,
                 f: float = 0.2):

        if N_wl % 2 == 0:
            raise ValueError("N_wl must be odd so a degenerate pair exists")

        self.x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
        self.dx = self.x[1] - self.x[0]

        # physical parameters
        self.wl0   = wl0
        self.Dwl   = Dwl
        self.N_wl  = N_wl
        self.wls   = self._get_wl_range()

        self.waist = waist
        self.x0    = x0
        self.f     = f
        self.blaze_angle = blaze_angle
        self.grating_mask = blazed_mask(self.x, self.blaze_angle, self.wl0)

        # Gaussian source (re-used for every field)
        self.E0 = gaussian(self.x, waist, x0)

        # reference detector axis (λ0)
        kx_ref = fftshift(fftfreq(Nx, d=self.dx) * 2 * np.pi)
        # TODO: maybe should be self.wls[-1]? I want the field with smallest X, so it will be only interpolation
        self.x_det_ref = f * self.wls[0] * kx_ref / (2 * np.pi)

    # -----------------------------------------------------------------
    def _get_wl_range(self):
        c = 299792458  # speed of light in m/s
        f0 = c / self.wl0
        frange = (c / self.wl0**2) * self.Dwl
        df = frange / self.N_wl
        f = f0 + np.arange(-self.N_wl / 2, self.N_wl / 2) * df
        l = c / f
        return l[::-1]


    def classical_pattern(self) -> tuple[np.ndarray, np.ndarray]:
        I_tot = np.zeros_like(self.x_det_ref)

        for wl in self.wls:
            E = self.E0.copy()
            E *= self.grating_mask * self.wl0/wl  # TODO: dispersion
            # E *= lens_phase(self.x, wl, self.f)
            x_det, I = farfield(E, wl, self.f, self.dx)
            I_tot += regrid(I, x_det, self.x_det_ref)

        I_tot /= I_tot.max()
        return self.x_det_ref, I_tot

    def spdc_pattern(self) -> tuple[np.ndarray, np.ndarray]:
        I_tot = np.zeros_like(self.x_det_ref)

        for i, wl_s in enumerate(self.wls):
            wl_i = self.wls[self.N_wl - 1 - i]          # mirror pairing

            E = self.E0.copy()
            E *= self.grating_mask * self.wl0 / wl_s  # TODO: dispersion
            # TODO: phase matching
            E *= self.grating_mask * self.wl0 / wl_i  # TODO: dispersion

            # E *= lens_phase(self.x, wl_i, self.f)       # quadratic phase

            x_det, I = farfield(E, wl_i, self.f, self.dx)
            I_tot += regrid(I, x_det, self.x_det_ref)

        I_tot /= I_tot.max()
        return self.x_det_ref, I_tot
