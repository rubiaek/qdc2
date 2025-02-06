import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from qdc.fiber import Fiber

class ManyWavelengthFiber(object):
    def __init__(self, wl0=0.810, Dwl=0.040, N_wl=81, fiber_L=2e6):
        """
        Creates a list of Fiber objects across a range of wavelengths.
        They can share the same length, etc., but each has its own solved modes.

        wl0, Dwl, N_wl: define the spectral range
        fiber_L: length in microns
        """
        self.wl0 = wl0
        self.Dwl = Dwl
        self.N_wl = N_wl
        self.wls = self._get_wl_range()
        self.ns = self._sellmeier_silica(self.wls)
        self.fibers = []

        print(f"Getting {N_wl} fibers...")
        for i, wl in tqdm(enumerate(self.wls)):
            self.fibers.append(Fiber(wl=wl, n1=self.ns[i], L=fiber_L))
        print("Got fibers!")

        # In case the number of modes differs slightly at the edges of the spectrum, we
        # define a safe "cutoff" for the # of modes we can use across all wls.
        self.N_modes_cutoff = min(self.fibers[0].Nmodes, self.fibers[-1].Nmodes)
        self.betas = np.zeros((N_wl, self.N_modes_cutoff))
        self._populate_betas()

    def _populate_betas(self):
        for i, f in enumerate(self.fibers):
            self.betas[i, :] = f.modes.betas[: self.N_modes_cutoff]

    def _sellmeier_silica(self, wls):
        """
        Sellmeier equation for fused silica: returns n vs. wavelength (in microns).
        """
        a1 = 0.6961663
        a2 = 0.4079426
        a3 = 0.8974794
        b1 = 0.0684043
        b2 = 0.1162414
        b3 = 9.896161
        return np.sqrt(
            1
            + a1 * (wls**2) / (wls**2 - b1**2)
            + a2 * (wls**2) / (wls**2 - b2**2)
            + a3 * (wls**2) / (wls**2 - b3**2)
        )

    def _get_wl_range(self):
        """
        Return array of N_wl wavelengths spanning +/- Dwl/2 around wl0.
        """
        c = 299792458e6  # speed of light in um/s
        f0 = c / self.wl0
        frange = (c / self.wl0**2) * self.Dwl
        df = frange / self.N_wl
        f = f0 + np.arange(-self.N_wl / 2, self.N_wl / 2) * df
        l = c / f
        return l[::-1]

    def set_inputs_gaussian(self, sigma=10, X0=3, Y0=9, X_linphase=0.3, random_phase=0.5):
        """
        Convenience method: set a Gaussian profile input for each fiber (each wavelength).
        """
        for f in self.fibers:
            f.set_input_gaussian(
                sigma=sigma,
                X0=X0,
                Y0=Y0,
                X_linphase=X_linphase,
                random_phase=random_phase,
            )

    def set_inputs_random_modes(self, N_random_modes=30):
        """
        Convenience method: set random superposition of modes for each fiber.
        """
        for f in self.fibers:
            f.set_input_random_modes(N_random_modes)

    def propagate_free_space(self, E, dz, fiber):
        """
        Propagate a 2D field E by distance dz in free space using FFT.
        The parameter 'fiber' is used for pixel size (dh) and wavelength (wl).

        Returns flattened field (raveled).
        """
        if E.ndim == 1:
            n = int(np.sqrt(E.size))
            E = E.reshape([n, n])

        fa = np.fft.fft2(E)
        dx = fiber.index_profile.dh
        dy = fiber.index_profile.dh
        freq_x = np.fft.fftfreq(E.shape[1], d=dx)
        freq_y = np.fft.fftfreq(E.shape[0], d=dy)
        freq_Xs, freq_Ys = np.meshgrid(freq_x, freq_y)

        light_k = 2 * np.pi / fiber.wl
        k_x = freq_Xs * 2 * np.pi
        k_y = freq_Ys * 2 * np.pi

        k_z_sqr = light_k**2 - (k_x**2 + k_y**2)
        # Remove negative => evanescent
        np.maximum(k_z_sqr, 0, out=k_z_sqr)
        k_z = np.sqrt(k_z_sqr)

        # Phase factor for free-space propagation
        fa *= np.exp(1j * k_z * dz)

        out_E = np.fft.ifft2(fa)
        return out_E.ravel()
