from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------
try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    fft = pyfftw.interfaces.numpy_fft.fft
except ModuleNotFoundError:
    from numpy.fft import fft

from numpy.fft import fftshift, fftfreq


# ---------------------------------------------------------------------
def gaussian_field(x: np.ndarray, waist: float, x0: float = 0.0) -> np.ndarray:
    """Normalised 1-D Gaussian."""
    E = np.exp(-((x - x0) / waist) ** 2)
    return E / np.linalg.norm(E)


def lens_phase(x: np.ndarray, wl: float, f: float) -> np.ndarray:
    """Paraxial quadratic phase of a thin lens."""
    k = 2 * np.pi / wl
    return np.exp(-1j * k * x**2 / (2 * f))


def blazed_mask(x: np.ndarray,
                wl_ref: float,
                blaze_angle: float,
                wl: float) -> np.ndarray:
    """
    Ideal 2π saw-tooth reflection grating.

    At wl_ref the phase rises by exactly 2π over one period d.
    Detuning to wl multiplies the slope by wl_ref / wl.
    """
    d = wl_ref / np.sin(blaze_angle)        # groove period
    phase = -2 * np.pi * (wl_ref / wl) * x / d
    return np.exp(1j * phase)


# ---------------------------------------------------------------------
class GratingSim1D:
    """
    1-D blazed-grating simulation (FFT version).

    Parameters
    ----------
    Nx          : number of grid points
    Lx          : total simulation window [m]
    wl0         : reference wavelength for the blaze [m]
    Dwl         : total spectral span for classical sum [m]
    N_wl        : number of discrete wavelengths (odd number)
    waist       : 1/e Gaussian waist radius in the grating plane [m]
    x0          : centre of the Gaussian beam [m]
    blaze_deg   : blaze angle in degrees   (d·sinθ = wl0)
    f           : focal length of the lens [m]
    """

    def __init__(self,
                 Nx: int = 2**14,
                 Lx: float = 8e-3,
                 wl0: float = 808e-9,
                 Dwl: float = 100e-9,
                 N_wl: int = 11,
                 waist: float = 40e-6,
                 x0: float = 0.0,
                 blaze_deg: float = 0.15,
                 f: float = 0.2):

        if N_wl % 2 == 0:
            raise ValueError("N_wl must be odd so that a degenerate pair exists")

        self.x = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False)
        self.dx = self.x[1] - self.x[0]

        self.wl0 = wl0
        self.Dwl = Dwl
        self.N_wl = N_wl
        self.wls = self._make_wavelength_array()

        self.waist = waist
        self.x0 = x0
        self.blaze_angle = np.deg2rad(blaze_deg)
        self.f = f

    # -----------------------------------------------------------------
    def _make_wavelength_array(self) -> np.ndarray:
        """Return `N_wl` wavelengths equally spaced in frequency around wl0."""
        c = 299_792_458.0
        f0 = c / self.wl0
        df = c / (self.wl0**2) * self.Dwl / (self.N_wl - 1)
        freqs = f0 + (np.arange(self.N_wl) - (self.N_wl - 1)/2) * df
        return c / freqs

    # -----------------------------------------------------------------
    def _farfield_intensity(self, field: np.ndarray) -> np.ndarray:
        """FFT, centre zero-freq, return |E|²."""
        Ef = fftshift(fft(fftshift(field)))
        return np.abs(Ef)**2

    def classical_farfield(self, wl: float) -> np.ndarray:
        E = gaussian_field(self.x, self.waist, self.x0).astype(np.complex128)
        E *= blazed_mask(self.x, self.wl0, self.blaze_angle, wl)
        E *= lens_phase(self.x, wl, self.f)
        return self._farfield_intensity(E)

    def spdc_farfield(self, wl_plus: float, wl_minus: float) -> np.ndarray:
        E = gaussian_field(self.x, self.waist, self.x0).astype(np.complex128)
        E *= blazed_mask(self.x, self.wl0, self.blaze_angle, wl_plus)
        E *= blazed_mask(self.x, self.wl0, self.blaze_angle, wl_minus)
        E *= lens_phase(self.x, wl_minus, self.f)
        return self._farfield_intensity(E)

    # -----------------------------------------------------------------
    def run(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (classical incoherent sum, SPDC coincidence sum)."""
        I_classical = sum(self.classical_farfield(wl) for wl in self.wls)

        mid = self.N_wl // 2
        I_spdc = self.spdc_farfield(self.wls[mid], self.wls[mid])  # degenerate
        for i in range(1, mid + 1):
            wl_plus, wl_minus = self.wls[mid + i], self.wls[mid - i]
            I_spdc += self.spdc_farfield(wl_plus, wl_minus)

        return I_classical, I_spdc
