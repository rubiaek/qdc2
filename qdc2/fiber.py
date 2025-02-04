import os
import numpy as np
import matplotlib.pyplot as plt
from pianoq.misc.misc import Player, colorize
import cv2
import pyMMF
from pyMMF.modes import Modes
import logging
logging.disable()

MODES_DIR = "C:\\temp\\MMF_modes"


SOLVER_N_POINTS_SEARCH = 2**8
SOLVER_N_POINTS_MODE = 2**7
SOLVER_R_MAX_COEFF = 1.8
SOLVER_BC_RADIUS_STEP = 0.95
SOLVER_N_BETA_COARSE = 1000
SOLVER_MIN_RADIUS_BC = .5


class Fiber(object):
    def __init__(self, wl=0.808, n1=1.453, NA=0.2, diameter=50, curvature=None, areaSize=None, npoints=2**7, autosolve=True, L=5e6):
        """ all in um """
        self.rng = np.random.default_rng(12345)

        self.NA = NA
        self.diameter = diameter
        self.radius = self.diameter / 2  # in microns
        self.areaSize = areaSize or 2.5*self.radius  # calculate the field on an area larger than the diameter of the fiber
        self.npoints = npoints  # resolution of the window
        self.n1 = n1
        self.wl = wl  # wavelength in microns
        self.curvature = curvature
        self.L = L

        self.index_profile = pyMMF.IndexProfile(npoints=npoints, areaSize=self.areaSize)
        self.index_profile.initParabolicGRIN(n1=n1, a=self.radius, NA=NA)

        self.solver = pyMMF.propagationModeSolver()
        self.solver.setIndexProfile(self.index_profile)
        self.solver.setWL(self.wl)

        self.profile_0 = np.zeros(self.npoints**2)
        self.modes_0 = None
        self.profile_end = np.zeros(self.npoints**2)
        self.modes_end = None

        self.NmodesMax = pyMMF.estimateNumModesGRIN(self.wl, self.radius, self.NA)
        self.modes = None
        self.Nmodes = None
        self.saveto_path = os.path.join(MODES_DIR, f'modes_GRIN_wl={self.wl}_npoints={self.npoints}.npz')

        if autosolve:
            self.solve()

    def solve(self):
        if self._load_from_file():
            return
        r_max = SOLVER_R_MAX_COEFF * self.diameter
        k0 = 2 * np.pi / self.wl
        dh = self.diameter / SOLVER_N_POINTS_SEARCH
        mode_repr = 'cos'  # or exp for OAM modes

        self.modes = self.solver.solve(mode='radial_test',
                                       r_max=r_max,  # max radius to calculate (and first try for large radial boundary condition)
                                       dh=dh,  # radial resolution during the computation
                                       min_radius_bc=SOLVER_MIN_RADIUS_BC,  # min large radial boundary condition
                                       change_bc_radius_step=SOLVER_BC_RADIUS_STEP,  # change of the large radial boundary condition if fails
                                       N_beta_coarse=SOLVER_N_BETA_COARSE,  # number of steps of the initial coarse scan
                                       degenerate_mode=mode_repr,
                                       field_limit_tol=1e-4,)
        self.Nmodes = self.modes.number
        self._save_to_file()

    def _load_from_file(self):
        if not os.path.exists(self.saveto_path):
            return False

        f = open(self.saveto_path, 'rb')
        data = np.load(f, allow_pickle=True)

        self.modes = Modes()
        self.modes.number = data['n_modes']
        self.npoints = data['npoints']
        self.modes.modeMatrix = data['profiles']
        self.modes.betas = data['betas']
        self.modes.wl = self.wl
        assert (self.index_profile.n == data['index_profile_n']).all()
        self.modes.indexProfile = self.index_profile
        self.modes.profiles = list(self.modes.modeMatrix.T)
        self.Nmodes = self.modes.number

        return True

    def _save_to_file(self):
        f = open(self.saveto_path, 'wb')

        np.savez(
            f,
            n_points=SOLVER_N_POINTS_MODE,
            n_modes=self.Nmodes,
            npoints=self.npoints,
            profiles=self.modes.getModeMatrix(),
            betas=self.modes.betas,
            index_profile_n=self.index_profile.n)

        f.close()

    def _get_gausian(self, sig, X0=0, Y0=0, X_linphase=0.0, Y_linphase=0.0, random_phase=0.0, ravel=True):
        """ sig in pixels """
        X = np.arange(-self.npoints/2, self.npoints/2)
        XX, YY = np.meshgrid(X, X)
        # sqrt in 2d, and also 4*sig**2, because field and not power, so abs(g)**2.sum() = 1
        g = 1 / np.sqrt(sig**2 * 2 * np.pi) * np.exp(-((XX-X0) ** 2 + (YY-Y0) ** 2) / (4 * sig ** 2))

        if X_linphase != 0 or Y_linphase != 0:
            g = np.exp(1j*(XX*X_linphase + YY*Y_linphase)) * g

        if random_phase != 0:
            A = random_phase*self.rng.normal(size=(40, 40))
            A = cv2.resize(A, g.shape, interpolation=cv2.INTER_AREA)
            g *= np.exp(1j*A)

        if ravel:
            return g.ravel()
        else:
            return g

    def set_input_gaussian(self, sigma=10, X0=0, Y0=0, X_linphase=0.0, Y_linphase=0.0, random_phase=0.0):
        self.profile_0 = self._get_gausian(sig=sigma, X0=X0, Y0=Y0,
                                           X_linphase=X_linphase, Y_linphase=Y_linphase,
                                           random_phase=random_phase, ravel=True)

    def set_input_random_modes(self, first_N_modes=50):
        amps = np.zeros(self.Nmodes).astype(np.complex128)
        amps[:first_N_modes] = self.rng.uniform(-1, 1, first_N_modes) + 1j * self.rng.uniform(-1, 1, first_N_modes)
        C = np.sqrt(((np.abs(amps)**2).sum()))
        self.modes_0 = amps / C
        self.profile_0 = self.modes_0.T @ self.modes.getModeMatrix().T

    def propagate(self, show=True):
        self.modes_0 = self.modes.getModeMatrix().T @ self.profile_0
        # evolute modes
        self.modes_end = self.modes.getPropagationMatrix(distance=self.L) @ self.modes_0
        # modes to profile (not shifted, because it is what it is)
        self.profile_end = self.modes_end.T @ self.modes.getModeMatrix().T

        if show:
            fig, ax = plt.subplots(1, 2)
            self.show_profile(self.profile_0, ax[0], title='profile_0')
            self.show_profile(self.profile_end, ax[1], title='profile_end')
            fig.show()

        return self.profile_end

    def show_profile(self, profile, ax=None, title=''):
        if len(profile.shape) == 1:
            n = np.sqrt(profile.size)
            assert n.is_integer()
            n = int(n)
            profile = profile.reshape([n] * 2)

        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(colorize(profile))
        power_transmitted = (np.abs(profile)**2).sum()
        ax.set_title(f'total power: {power_transmitted:.3f}, {title}')
        ax.figure.show()

    def show_mode(self, m):
        """ m mode number"""
        fig, axes = plt.subplots(2)
        axes[0].imshow(np.real(self.modes.getModeMatrix()[:, m]).reshape([self.npoints]*2))
        axes[1].imshow(np.imag(self.modes.getModeMatrix()[:, m]).reshape([self.npoints]*2))
        fig.show()

    def animate_modes(self):
        fig = plt.figure(figsize = (10,5))
        ax = fig.add_subplot(1,1,1)
        ax.set_ylim(0, 1)
        palette = ['blue', 'red', 'green',
                   'darkorange', 'maroon', 'black']

        def animation_function(i):
            ax.clear()
            self.show_profile(self.modes.getModeMatrix()[:, i], ax=ax)
            ax.set_title(f'mode num: {i}')

        animation = Player(fig, animation_function, interval=500, frames=self.Nmodes)
        plt.show()


"""
from pianoq.simulations.qdc2.fiber import Fiber
f = Fiber(L=2e5)
f.set_input_gaussian(sigma=7, X0=25, Y0=-25); 
f.propagate()

"""