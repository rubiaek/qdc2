import numpy as np
from tqdm import tqdm
from qdc.mmf.fiber import Fiber

class ManyWavelengthFiber(object):
    def __init__(self, wl0=0.810, Dwl=0.040, N_wl=81, fiber_L=2e6, rng_seed=12345, is_step_index=False, 
                 npoints=2**7, NA_ref=0.2, autosolve=True):
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
        self.ns_clad = self._sellmeier_silica(self.wls)
        self.fibers = []
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(self.rng_seed)
        self.is_step_index = is_step_index
        self.NA_ref = NA_ref
        n_clad_ref  = self._sellmeier_silica(np.array([self.wl0]))[0]
        # NA = sqrt(n_core^2 - n_clad^2)
        # delta_n = n_core - n_clad
        self.delta_n = np.sqrt(n_clad_ref**2 + NA_ref**2) - n_clad_ref
        self.autosolve = autosolve


        self.gaussian_params = np.array([7, 7, 7, 0.4, 0.4])  # sigma, X0, Y0, X_linphase, Y_linphase
        self.gaussian_dparams = np.array([0, 1, 1, 0.1, 0.1])  # sigma, X0, Y0, X_linphase, Y_linphase
            # {'sigma': 7, 'X0': 7, 'Y0': 7, 'X_linphase': 0.4, 'Y_linphase': 0.4}

        print(f"Getting {N_wl} fibers...")
        for i, wl in tqdm(enumerate(self.wls)):
            n_core = self.ns_clad[i] + self.delta_n
            NA_i   = np.sqrt(n_core**2 - self.ns_clad[i]**2)
            f = Fiber(wl=wl, n1=n_core, NA=NA_i, L=fiber_L, rng_seed=rng_seed, is_step_index=self.is_step_index, 
                      npoints=npoints, autosolve=self.autosolve)
            self.fibers.append(f)
        print("Got fibers!")

        self.dx = self.fibers[0].index_profile.dh
        for f in self.fibers:
            eps = abs(f.index_profile.dh - self.dx)
            assert eps < 1e-12, "All fibers must have the same dx!"

        N_beta_cutoff = min(len(f.modes.betas) for f in self.fibers)
        self.betas = np.array([f.modes.betas[:N_beta_cutoff] for f in self.fibers])

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

    def get_random_dparams(self):
        sig, x, y, xp, yp = self.gaussian_dparams

        dsigma = self.rng.uniform(low=-sig, high=sig)
        dX0 = self.rng.uniform(low=-x, high=x)
        dY0 = self.rng.uniform(low=-y, high=y)
        dX_linphase = self.rng.uniform(low=-xp, high=xp)
        dY_linphase = self.rng.uniform(low=-yp, high=yp)
        return np.array([dsigma, dX0, dY0, dX_linphase, dY_linphase])

    def get_g_params(self, add_random=True):
        params = self.gaussian_params.copy()
        if add_random:
            params += self.get_random_dparams()
        return params

    def set_inputs_random_modes(self, N_random_modes=30):
        for f in self.fibers:
            f.set_input_random_modes(N_random_modes)
