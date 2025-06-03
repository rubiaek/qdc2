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


def blazed_phase(x, theta, wl_ref) -> np.ndarray:
    d = wl_ref / np.sin(theta)  # grating period, d*sin(theta)=lambda*m; m=1
    phi = (2 * np.pi / d * x) % (2 * np.pi)
    return phi


def farfield(E, wl, f, dx):
    N = len(E)
    Ef = fftshift(fft(fftshift(E))) * dx          # include sampling
    I  = np.abs(Ef) ** 2

    kx = fftshift(fftfreq(N, d=dx) * 2 * np.pi)
    x_det = f * wl * kx / (2 * np.pi)

    # energy normalisation
    I /= np.trapz(I, x_det)

    return x_det, I


def regrid(I_src, x_src, x_ref):
    """Interpolate I(x_src) → I(x_ref), zero outside range."""
    return np.interp(x_ref, x_src, I_src, left=0.0, right=0.0)


class GratingSim1D:
    """
    1-D Blazed-Grating Simulator

    Parameters
    ----------
    Nx, Lx       : grid size and window [m]
    wl0          : central blaze wavelength [m]
    Dwl          : full spectral span [m]
    N_wl         : number of wavelength samples (odd!)
    waist, x0    : Gaussian waist [m] and centre [m]
    blaze_angle  : blaze angle [rad]
    f            : lens focal length [m]
    spectrum     : 'flat' or 'gaussian' spectral weighting
    spec_sigma   : σ for Gaussian spectrum [m] (defaults to Dwl/2)
    """
    def __init__(self,
        Nx: int        = 2**14,
        Lx: float      = 8e-3,
        wl0: float     = 808e-9,
        Dwl: float     = 100e-9,
        N_wl: int      = 31,
        waist: float   = 800e-6,
        x0: float      = 0.0,
        blaze_angle: float = 0.15,
        f: float       = 0.2,
        spectrum: str  = 'flat',
        spec_sigma: float | None = None
    ):
        if N_wl % 2 == 0:
            raise ValueError("N_wl must be odd for a degenerate pair")

        # real‐space grid
        self.x  = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
        self.dx = self.x[1] - self.x[0]

        # wavelengths
        self.wl0  = wl0
        self.Dwl  = Dwl
        self.N_wl = N_wl
        self.wls  = self._get_wl_range()

        # beam + optics
        self.waist = waist
        self.x0    = x0
        self.f     = f
        self.blaze = blaze_angle

        # spectral weighting
        self.spectrum   = spectrum
        self.spec_sigma = spec_sigma or (Dwl/2)
        self._make_weights()

        # grating phase (for λ0)
        self.grating_phase = blazed_phase(self.x, self.blaze, self.wl0)

        # Gaussian source
        self.E0 = gaussian(self.x, waist, x0)

        # reference detector axis (for λ0)
        kx_ref = fftshift(fftfreq(Nx, d=self.dx)*2*np.pi)
        self.x_det_ref = f * wl0 * kx_ref/(2*np.pi)

    def _get_wl_range(self):
        if self.N_wl == 1:
            return np.array([self.wl0])
            
        c = 299792458  # speed of light in m/s
        f0 = c / self.wl0  # center frequency
        d_lambda = self.Dwl / (self.N_wl - 1)
        df = (c / self.wl0**2) * d_lambda  # df/d_lambda=c/lambda^2;  
        
        # Generate frequencies symmetrically around f0
        n = (self.N_wl - 1) // 2
        f = f0 + np.arange(-n, n+1) * df
        
        # Convert to wavelengths, longest to shortest
        return (c / f)[::-1]

    def _make_weights(self):
        """Set self.weights[i] per wavelength based on spectrum shape."""
        if self.spectrum=='flat':
            self.weights = np.ones(self.N_wl)
        else:  # gaussian
            eps = (self.wls - self.wl0)/self.spec_sigma
            w   = np.exp(-0.5*eps**2)
            self.weights = w/np.sum(w)
        
        self.weights /= self.weights.sum()

    def classical_pattern(self):
        """Return x_det_ref, I_classical (max=1)."""
        I_tot = np.zeros_like(self.x_det_ref)
        for w_i, wl in zip(self.weights, self.wls):
            E = self.E0.copy()
            E *= np.exp(1j*self.grating_phase*(self.wl0/wl))
            E *= lens_phase(self.x, wl, self.f)
            x_det, I = farfield(E, wl, self.f, self.dx)
            I_tot += w_i * regrid(I, x_det, self.x_det_ref)
        I_tot /= I_tot.max()
        return self.x_det_ref, I_tot

    def spdc_pattern(self):
        """Return x_det_ref, I_spdc (max=1)."""
        I_tot = np.zeros_like(self.x_det_ref)
        for w_i, wl_s in zip(self.weights, self.wls):
            wl_i = self.wls[self.N_wl-1 - self.wls.tolist().index(wl_s)]
            E = self.E0.copy()
            E *= np.exp(1j*self.grating_phase*(self.wl0/wl_s)) # TODO: dispersion
            # TODO: phase matching
            E *= np.exp(1j*self.grating_phase*(self.wl0/wl_i))# TODO: dispersion
            E *= lens_phase(self.x, wl_i, self.f)
            x_det, I = farfield(E, wl_i, self.f, self.dx)
            I_tot += w_i * regrid(I, x_det, self.x_det_ref)
        I_tot /= I_tot.max()
        return self.x_det_ref, I_tot

    def diffraction_orders(self, x_det):
        """Map x_det→ diffraction order m = (x′/f)*(d/λ₀)."""
        d = self.wl0/np.sin(self.blaze)
        return (x_det/self.f)*(d/self.wl0)
