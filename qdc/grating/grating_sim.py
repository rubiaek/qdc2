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
    g = np.exp(-((x - x0) / w) ** 2)
    g /= np.sqrt((np.abs(g)**2).sum())
    return g.astype(np.complex128)


def lens_phase(x, wl, f) -> np.ndarray:
    """Thin-lens quadratic phase  exp(-iπ x² / (λ f))."""
    return np.exp(-1j * np.pi * x ** 2 / (wl * f))


def blazed_phase(x, d) -> np.ndarray:
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
        self.k0   = 2 * np.pi / self.wl0
        self.Dwl  = Dwl
        self.N_wl = N_wl
        self.wls  = self._get_wl_range()

        # beam + optics
        self.waist_at_wl0 = waist
        self.x0    = x0
        self.f     = f
        self.blaze = blaze_angle
        self.d = self.wl0/np.sin(self.blaze) # grating period, d*sin(theta)=lambda*m; m=1

        # spectral weighting
        self.spectrum   = spectrum
        self.spec_sigma = spec_sigma or (Dwl/2)
        self._make_spectral_weights()

        self.grating_phase = blazed_phase(self.x, self.d)

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

    def _make_spectral_weights(self):
        """Set self.weights[i] per wavelength based on spectrum shape."""
        if self.spectrum=='flat':
            self.spectral_weights = np.ones(self.N_wl)
        else:  # gaussian
            eps = (self.wls - self.wl0)/self.spec_sigma
            w   = np.exp(-0.5*eps**2)
            self.spectral_weights = w/np.sum(w)
        
        self.spectral_weights /= self.spectral_weights.sum()

    def classical_pattern(self):
        """Return x_det_ref, I_classical."""
        I_tot = np.zeros_like(self.x_det_ref)
        for w_i, wl in zip(self.spectral_weights, self.wls):
            E = gaussian(self.x, self.waist_at_wl0 * (wl / self.wl0), self.x0)
            E *= np.exp(1j*self.grating_phase*(self.wl0/wl))
            E *= lens_phase(self.x, wl, self.f)
            x_det, I = farfield(E, wl, self.f, self.dx)
            I_tot += w_i * regrid(I, x_det, self.x_det_ref)
        

        return self.x_det_ref, I_tot
 
    def spdc_pattern(self):
        """Return x_det_ref, I_spdc."""
        I_tot = np.zeros_like(self.x_det_ref)
        for w_i, wl_s in zip(self.spectral_weights, self.wls):
            wl_i = self.wls[self.N_wl-1 - self.wls.tolist().index(wl_s)]
            # In the AWP we begin from an SMF with some MFD, and then different wavelengths 
            # will genereate sligthly different Gassuains at the farfield on the grating.
            E = gaussian(self.x, self.waist_at_wl0 * (wl_s / self.wl0), self.x0)
            E *= np.exp(1j*self.grating_phase*(self.wl0/wl_s)) # TODO: dispersion
            # TODO: phase matching
            E *= np.exp(1j*self.grating_phase*(self.wl0/wl_i))# TODO: dispersion
            E *= lens_phase(self.x, wl_i, self.f)
            x_det, I = farfield(E, wl_i, self.f, self.dx)
            I_tot += w_i * regrid(I, x_det, self.x_det_ref)

        return self.x_det_ref, I_tot

    def diffraction_orders(self, x_det):
        """Map x_det→ diffraction order m = (x′/f)*(d/λ₀)."""
        return (x_det/self.f)*(self.d/self.wl0)


    def analytical_pattern(self, *, is_spdc: bool, n_side: int = 5) -> tuple[np.ndarray, np.ndarray]:
        I_tot = np.zeros_like(self.x_det_ref)
                
        for lam, w_spec in zip(self.wls, self.spectral_weights):
            x_det = self.x_det_ref*lam/self.wl0 # bigger lam -> biger x_det (x in the farfield) 
            k = 2*np.pi/lam
            if is_spdc:
                orders = np.array([2])
                orders_kx = 2*np.pi*orders/self.d
                blaze_weights = self.d * np.sinc(orders - 2.0)
            else:
                m_centre = self.wl0 / lam
                orders = np.arange(int(m_centre - n_side), int(m_centre + n_side) + 1)
                orders_kx = 2*np.pi*orders/self.d
                blaze_weights = self.d * np.sinc(orders - m_centre)
            
            orders_x = self.f * orders_kx / k 

            E_tot = np.zeros_like(x_det, dtype=np.complex128)
            for m, blaze_weight, x_m in zip(orders, blaze_weights, orders_x):                
                real_waist_on_grating = self.waist_at_wl0 * (lam / self.wl0)
                this_waist_far_field_spot  = 2*self.f/(k*real_waist_on_grating)
                gaus_x = np.sqrt(np.pi) * real_waist_on_grating * gaussian(x_det, this_waist_far_field_spot, x_m)
                # 2pi/d is the Dirac comb prefactor, weight is the blaze weight, 
                E_tot += 2*np.pi/self.d * blaze_weight * gaus_x 

            E_tot /= np.sqrt(np.trapz(np.abs(E_tot)**2, x_det))
            I_tot_wl = w_spec * np.abs(E_tot)**2
            
            I_tot += regrid(I_tot_wl, x_det, self.x_det_ref) 
        
        return self.x_det_ref, I_tot