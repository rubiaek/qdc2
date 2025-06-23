import numpy as np
from tqdm import tqdm
from qdc.mmf.fiber import Fiber

class ManyWavelengthFiber(object):
    def __init__(self, wl0=0.810, Dwl=0.040, N_wl=81, fiber_L=2e6, rng_seed=12345):
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
        self.rng_seed = rng_seed

        print(f"Getting {N_wl} fibers...")
        for i, wl in tqdm(enumerate(self.wls)):
            self.fibers.append(Fiber(wl=wl, n1=self.ns[i], L=fiber_L, rng_seed=rng_seed))
        print("Got fibers!")

        self.dx = self.fibers[0].index_profile.dh
        for f in self.fibers:
            eps = abs(f.index_profile.dh - self.dx)
            assert eps < 1e-12, "All fibers must have the same dx!"

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
        for f in self.fibers:
            f.set_input_gaussian(
                sigma=sigma,
                X0=X0,
                Y0=Y0,
                X_linphase=X_linphase,
                random_phase=random_phase,
            )

    def set_inputs_random_modes(self, N_random_modes=30):
        for f in self.fibers:
            f.set_input_random_modes(N_random_modes)
