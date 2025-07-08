import os
import numpy as np
import matplotlib.pyplot as plt
from qdc.misc import colorize
import cv2
import pyMMF
from pyMMF.modes import Modes
from importlib.metadata import version
version = version('pyMMF')
assert version.startswith("0.7"), 'Must use updated pyMMF, use pip install git+https://github.com/wavefrontshaping/pyMMF.git'
import logging
logging.disable()

MODES_DIR = "C:\\temp\\MMF_modes"

SOLVER_N_POINTS_SEARCH = 2**8
SOLVER_R_MAX_COEFF = 1.8
SOLVER_BC_RADIUS_STEP = 0.95
SOLVER_N_BETA_COARSE = 1000
SOLVER_MIN_RADIUS_BC = .5

class Fiber(object):
    def __init__(
        self,
        wl=0.808,
        n1=1.453,
        NA=0.2,
        diameter=50,
        curvature=None,
        areaSize=None,
        npoints=2**7,
        autosolve=True,
        L=5e6,
        rng_seed=12345,
        is_step_index=False
    ):
        """
        Single-wavelength fiber model using pyMMF.

        Example usage:
            from qdc.fiber import Fiber
            f = Fiber(L=0.2e6)
            f.set_input_gaussian(sigma=7, X0=25, Y0=-25);
            f.propagate()
        """
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(self.rng_seed)

        self.NA = NA
        self.diameter = diameter
        self.radius = self.diameter / 2  # in microns
        self.areaSize = areaSize or 2.5 * self.radius  # calculate the field on an area larger than the diameter of the fiber
        self.npoints = npoints  # resolution of the window
        self.n1 = n1
        self.wl = wl  # wavelength in microns
        self.curvature = curvature
        self.L = L
        self.is_step_index = is_step_index

        # Setup pyMMF index profile
        self.index_profile = pyMMF.IndexProfile(npoints=npoints, areaSize=self.areaSize)
        self.dx = self.index_profile.dh
        if self.is_step_index:
            self.index_profile.initStepIndex(n1=n1, a=self.radius, NA=NA)
        else:
            self.index_profile.initParabolicGRIN(n1=n1, a=self.radius, NA=NA)

        self.solver = pyMMF.propagationModeSolver()
        self.solver.setIndexProfile(self.index_profile)
        self.solver.setWL(self.wl)

        self.profile_0 = np.zeros(self.npoints**2, dtype=np.complex128)
        self.modes_0 = None
        self.profile_end = np.zeros(self.npoints**2, dtype=np.complex128)
        self.modes_end = None

        if self.is_step_index:
            self.NmodesMax = pyMMF.estimateNumModesSI(self.wl, self.radius, self.NA)
        else:
            self.NmodesMax = pyMMF.estimateNumModesGRIN(self.wl, self.radius, self.NA)
        self.modes = None
        self.Nmodes = None

        f_type = 'SI' if self.is_step_index else 'GRIN'

        self.saveto_path = os.path.join(
            MODES_DIR, f"modes_{f_type}_wl={self.wl:.4f}_npoints={self.npoints}.npz"
        )

        if autosolve:
            self.solve()

    def solve(self):
        """Solve for the modes, either by loading from file or by running the solver."""
        if self._load_from_file():
            return

        r_max = SOLVER_R_MAX_COEFF * self.diameter
        dh = self.diameter / SOLVER_N_POINTS_SEARCH
        mode_repr = "cos" if not self.is_step_index else 'sin'  # or "exp" for OAM modes

        options = {
            'degenerate_mode':mode_repr,
        }
        
        if not self.is_step_index:
            options['r_max'] = r_max
            options['dh'] = dh
            options['min_radius_bc'] = SOLVER_MIN_RADIUS_BC
            options['change_bc_radius_step'] = SOLVER_BC_RADIUS_STEP
            options['N_beta_coarse'] = SOLVER_N_BETA_COARSE
            options['field_limit_tol'] = 1e-4
            
        self.modes = self.solver.solve(
            solver="default",
            options=options
        )
        self.Nmodes = self.modes.number
        self._save_to_file()

    def _load_from_file(self):
        if not os.path.exists(self.saveto_path):
            return False
        with open(self.saveto_path, "rb") as f:
            data = np.load(f, allow_pickle=True)
            self.modes = Modes()
            self.modes.number = data["n_modes"]
            self.npoints = data["npoints"]
            self.modes.modeMatrix = data["profiles"]
            self.modes.betas = data["betas"]
            self.modes.wl = self.wl

            assert (self.index_profile.n == data["index_profile_n"]).all(), f"index_profile_n: {self.index_profile.n.shape}, data['index_profile_n']: {data['index_profile_n'].shape}"
            self.modes.indexProfile = self.index_profile
            self.modes.profiles = list(self.modes.modeMatrix.T)
            self.Nmodes = self.modes.number
        return True

    def _save_to_file(self):
        with open(self.saveto_path, "wb") as f:
            np.savez(
                f,
                n_points=self.npoints,
                n_modes=self.Nmodes,
                npoints=self.npoints,
                profiles=self.modes.getModeMatrix(),
                betas=self.modes.betas,
                index_profile_n=self.index_profile.n,
            )

    def _get_gausian(
        self, sig, X0=0, Y0=0, X_linphase=0.0, Y_linphase=0.0, random_phase=0.0, ravel=True
    ):
        """sig in microns."""
        X = np.linspace(-self.areaSize/2, self.areaSize/2, self.npoints)
        YY, XX = np.meshgrid(X, X)
        g = 1 / np.sqrt(sig**2 * 2 * np.pi) * np.exp(-((XX - X0) ** 2 + (YY - Y0) ** 2) / (4 * sig**2))

        if X_linphase != 0 or Y_linphase != 0:
            g = np.exp(1j * (XX * X_linphase + YY * Y_linphase)) * g

        if random_phase != 0:
            A = random_phase * self.rng.normal(size=(40, 40))
            A = cv2.resize(A, g.shape, interpolation=cv2.INTER_AREA)
            g *= np.exp(1j * A)

        return g.ravel() if ravel else g

    def set_input_gaussian(
        self, sigma=10, X0=0, Y0=0, X_linphase=0.0, Y_linphase=0.0, random_phase=0.0
    ):
        self.profile_0 = self._get_gausian(
            sig=sigma,
            X0=X0,
            Y0=Y0,
            X_linphase=X_linphase,
            Y_linphase=Y_linphase,
            random_phase=random_phase,
            ravel=True,
        )

    def set_input_random_modes(self, first_N_modes=50):
        """Random superposition of fiber modes, as an alternative input."""
        amps = np.zeros(self.Nmodes, dtype=np.complex128)
        amps[:first_N_modes] = (
            self.rng.uniform(-1, 1, first_N_modes)
            + 1j * self.rng.uniform(-1, 1, first_N_modes)
        )
        C = np.sqrt(np.sum(np.abs(amps) ** 2))
        self.modes_0 = amps / C
        self.profile_0 = self.modes_0.T @ self.modes.getModeMatrix().T

    def propagate(self, show=True, free_mode_matrix=False):
        """Propagate the current self.profile_0 to self.profile_end using the fiber modes."""
        # Convert from real-space profile to mode coefficients

        if self.modes is None or self.modes.modeMatrix is None:
            self.solve()

        self.modes_0 = self.modes.getModeMatrix().T @ self.profile_0
        # Evolve mode amplitudes
        self.modes_end = self.modes.getPropagationMatrix(distance=self.L) @ self.modes_0
        # Convert back to real-space
        self.profile_end = self.modes_end.T @ self.modes.getModeMatrix().T

        if show:
            fig, ax = plt.subplots(1, 2)
            self.show_profile(self.profile_0, ax=ax[0], title="profile_0")
            self.show_profile(self.profile_end, ax=ax[1], title="profile_end")
            plt.show()

        if free_mode_matrix and self.modes is not None:
            self.modes.modeMatrix = None
            self.modes.profiles = None
            import gc
            gc.collect()

        return self.profile_end

    def show_profile(self, profile, ax=None, title=""):
        """Visualize a 2D complex field as a colorized image."""
        if profile.ndim == 1:
            n = int(np.sqrt(profile.size))
            profile = profile.reshape([n, n])

        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(colorize(profile))
        power_transmitted = (np.abs(profile) ** 2).sum()
        ax.set_title(f"total power: {power_transmitted:.3f}, {title}")

    def show_mode(self, m):
        """Show real and imaginary parts of one mode."""
        fig, axes = plt.subplots(2)
        n = self.npoints
        mode = self.modes.getModeMatrix()[:, m].reshape((n, n))
        axes[0].imshow(np.real(mode))
        axes[1].imshow(np.imag(mode))
        fig.show()