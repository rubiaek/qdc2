import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pianoq.simulations.dispersion_cancelation.fiber import Fiber


class ManyWavelengthFiber(object):
    def __init__(self, wl0=0.810, Dwl=0.040, N_wl=81, fiber_L=2e6):
        """ all in um """
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
        self.N_modes_cutoff = min(self.fibers[0].Nmodes, self.fibers[-1].Nmodes)  # If N_modes changes with wl - discard last modes
        self.betas = np.zeros((N_wl, self.N_modes_cutoff))
        self._populate_betas()

    def _populate_betas(self):
        for i, f in enumerate(self.fibers):
            self.betas[i, :] = f.modes.betas[:self.N_modes_cutoff]

    def _sellmeier_silica(self, wls):
        a1 = 0.6961663
        a2 = 0.4079426
        a3 = 0.8974794
        b1 = 0.0684043
        b2 = 0.1162414
        b3 = 9.896161

        ns_silica = np.sqrt(1 +
                      a1 * (wls**2) / (wls**2 - b1**2) +
                      a2 * (wls**2) / (wls**2 - b2**2) +
                      a3 * (wls**2) / (wls**2 - b3**2))
        return ns_silica

    def _get_wl_range(self):
        """ in um"""
        # stolen from Logan GMMNLSE-Solver-FINAL-master\solve_for_modes.m
        c = 299792458e6  # um/s
        f0 = c / self.wl0  # center frequency in THz
        frange = c / self.wl0**2 * self.Dwl
        df = frange / self.N_wl
        f = f0 + np.arange(-self.N_wl/2, self.N_wl/2)*df
        l = c / f  # um
        return l[::-1]

    def show_betas(self):
        fig, ax = plt.subplots()
        for f in self.fibers:
            ax.plot(f.modes.betas, label=f'$ \lambda= ${f.wl:.4f}')
            ax.set_xlabel('mode #')
            ax.set_ylabel(r'Propagation constant $\beta$ (in $\mu$m$^{-1}$)')
            ax.legend()
        fig.show()

        fig, ax = plt.subplots()
        for i in np.arange(0, 50, 5):
            ax.plot(self.wls, self.betas[:, i], label=f'mode no. {i}')
            ax.set_xlabel(r'$\lambda (\mu m m)$')
            ax.set_ylabel(r'Propagation constant $\beta$ (in $\mu$m$^{-1}$)')
            ax.legend()
        fig.show()

    def set_inputs_gaussian(self, sigma=10, X0=3, Y0=9, X_linphase=0.3, random_phase=0.5):
        for f in self.fibers:
            f.set_input_gaussian(sigma=sigma, X0=X0, Y0=Y0, X_linphase=X_linphase, random_phase=random_phase)

    def set_inputs_random_modes(self, N_random_modes=30):
        for f in self.fibers:
            self.fibers[0].set_input_random_modes(N_random_modes)

    def get_klyshko_PCCs(self, dz=0):
        """ dz is the error in imaging to crystal, so before swtiching wavelengths there is first some
         free space propagation """
        i_middle = len(self.fibers) // 2
        N_measurements = (len(self.fibers) // 2) + 1  # for 5 wls: 1 degenerate + 2 non degenerate
        pccs = np.zeros(N_measurements)
        delta_lambdas = np.zeros(N_measurements)
        self.fibers[i_middle].set_input_gaussian(sigma=10, X0=3, Y0=9, X_linphase=0.3, random_phase=0.5)

        # Simply propagate twice for degenerate
        f = self.fibers[i_middle]
        E_end0 = f.propagate(show=False)
        E_after_prop = self.propagate_dz(E_end0, dz, f)
        f.profile_0 = E_after_prop
        E_end0 = f.propagate(show=False)

        I_end0 = np.abs(E_end0) ** 2
        II0 = I_end0.reshape([self.fibers[i_middle].npoints] * 2)[50:80, 50:80]  # todo: better than 50:80
        pccs[0] = 1
        delta_lambdas[0] = 0

        for di in range(1, N_measurements):
            f_plus = self.fibers[i_middle+di]
            f_minus = self.fibers[i_middle-di]

            # first half fiber
            f_plus.set_input_gaussian(sigma=10, X0=3, Y0=9, X_linphase=0.3, random_phase=0.5)
            E_end = f_plus.propagate(show=False)

            # freespace
            E_after_half_freespace = self.propagate_dz(E_end, dz, f_plus)
            E_after_freespace = self.propagate_dz(E_after_half_freespace, dz, f_minus)

            # second half fiber
            f_minus.profile_0 = E_after_freespace
            E_end = f_minus.propagate(show=False)
            I_end = np.abs(E_end)**2
            II = I_end.reshape([self.fibers[i_middle].npoints] * 2)[50:80, 50:80]
            pccs[di] = np.corrcoef(II0.ravel(), II.ravel())[1, 0]
            delta_lambdas[di] = f_plus.wl - f_minus.wl

        return np.array(delta_lambdas), pccs

    def propagate_dz(self, E, dz, f):
        if len(E.shape) == 1:
            n = np.sqrt(E.size)
            assert n.is_integer()
            n = int(n)
            E = E.reshape([n] * 2)

        fa = np.fft.fft2(E)
        dx = f.index_profile.dh
        dy = f.index_profile.dh
        freq_x = np.fft.fftfreq(E.shape[1], d=dx)
        freq_y = np.fft.fftfreq(E.shape[0], d=dy)

        freq_Xs, freq_Ys = np.meshgrid(freq_x, freq_y)

        light_k = 2 * np.pi / f.wl
        k_x = freq_Xs * 2 * np.pi
        k_y = freq_Ys * 2 * np.pi

        k_z_sqr = light_k ** 2 - (k_x ** 2 + k_y ** 2)
        # Remove all the negative component, as they represent evanescent waves,
        # See Fourier Optics page 58
        np.maximum(k_z_sqr, 0, out=k_z_sqr)
        k_z = np.sqrt(k_z_sqr)

        # Propagate light by adding the phase,
        # see Fourier Optics page 74
        fa *= np.exp(1j * k_z * dz)

        out_E = np.fft.ifft2(fa)

        return out_E.ravel()

    def get_classical_PCCs(self):
        # i_middle = len(self.fibers) // 2
        # In classical case I just want the dependance for w0 going to one side.
        i_middle = 0
        pccs = np.zeros_like(self.fibers)

        E_end0 = self.fibers[i_middle].propagate(show=False)
        I_end0 = np.abs(E_end0) ** 2
        II0 = I_end0.reshape([self.fibers[i_middle].npoints] * 2)[50:80, 50:80]  # todo: better than 50:80

        for i, f in enumerate(self.fibers):
            E_end = f.propagate(show=False)
            I_end = np.abs(E_end)**2

            II = I_end.reshape([self.fibers[i_middle].npoints] * 2)[50:80, 50:80]
            pccs[i] = np.corrcoef(II0.ravel(), II.ravel())[1, 0]

        delta_lambdas = [f.wl - self.fibers[i_middle].wl for f in self.fibers]

        return np.array(delta_lambdas), pccs

    def get_classical_PCCs_average(self, N_configs=5):
        pccs = np.zeros((N_configs, len(self.fibers)))
        delta_lambdas = np.zeros(len(self.fibers))
        for i in tqdm(range(N_configs)):
            self.set_inputs_gaussian()
            delta_lambdas, pccs[i, :] = self.get_classical_PCCs()
        return delta_lambdas, pccs.mean(axis=0)

    def get_klyshko_PCCs_average(self, N_configs=5, dz=0):
        pccs = np.zeros((N_configs, 1 + len(self.fibers) // 2))
        delta_lambdas = np.zeros(len(self.fibers))
        for i in tqdm(range(N_configs)):
            delta_lambdas, pccs[i, :] = self.get_klyshko_PCCs(dz=dz)
        return delta_lambdas, pccs.mean(axis=0)

    def show_PCC_classical_and_quantum(self, delta_lambdas_classical, pccs_classical, delta_lambdas_klyshko, pccs_klyshko, fiber_L, mode_mixing, dz):
        fig, ax = plt.subplots()
        ax.plot(delta_lambdas_classical * 1e3, pccs_classical, label='classical')
        ax.plot(delta_lambdas_klyshko * 1e3, pccs_klyshko, label='Klyshko')
        ax.set_xlabel(r'wl difference $ \Delta\lambda$ (nm)')
        ax.set_ylabel(r'PCC')
        ax.legend()
        ax.set_title(f'L: {fiber_L*1e-6}m, mode mixing: {0}, dz: {dz}um')
        fig.show()

    def run_PCCs_different_dz(self, dzs=(0, 20, 40, 60, 80), N_classical=5, N_klyshko=2):
        fig, ax = plt.subplots()

        print(f'Getting classical with average on {N_classical}...')
        delta_lambdas_classical, pccs_classical = self.get_classical_PCCs_average(N_classical)
        ax.plot(delta_lambdas_classical * 1e3, pccs_classical, label='classical', linewidth=3)

        for dz in dzs:
            print(f'Getting Klyshko with average on {N_klyshko} dz={dz}......')
            delta_lambdas_klyshko, pccs_klyshko = self.get_klyshko_PCCs_average(N_klyshko, dz=dz)
            ax.plot(delta_lambdas_klyshko * 1e3, pccs_klyshko, label=f'Klyshko dz={dz}$\mu m$')

        ax.set_xlabel(r'wl difference $ \Delta\lambda$ (nm)')
        ax.set_ylabel(r'PCC')
        ax.legend()
        ax.set_title(f'L: {self.fibers[0].L*1e-6}m')  # mode mixing: {0}
        fig.show()

    # find length that will cause this fiber to have a spectral correlation width of ~3nm, and then check our Kilshko two-photon spectral correlation width
    # this can be defined via choosing a length L such that max_m{beta_m(w+)-beta_m(w-)}*L = 2*pi for w+ - w- = 3nm
    # then check for same L the (w+ - w-) value such that max_m{beta_m(w+)+beta_m(w-)-w*beta_m(w0)}*2*L = 2*pi, and hope this is larger than 3nm
    # def spectral_correlation_width(self):
    #     Dbetas = s.betas[10, :] - s.betas[13, :]  # take 3nm of spectrum
    #     Dbetas -= np.median(Dbetas) # Global phase. median and not mean because there are outliers
    #     everything here is in microns, so 2m of piano is 2e6,
        # L = 2e6  # 2m fiber
        # this gives Dbetas~1 at least for the first ~30 modes, except for a few modes specific modes, not sure why
        # Dphis = Dbetas*L
        #
        # DDbetas = s.betas[0, :] + s.betas[20, :] - 2 * s.betas[10, :]  # Klyshko picture, 20nms hoping is OK
        # DDbetas -= np.median(DDbetas)
        # DDphis = DDbetas*2*L  # seems good! even with 20nm bandwidth we seem to still be far off from 2pi!

