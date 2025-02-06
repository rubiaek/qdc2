import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from qdc.many_wl_fiber import ManyWavelengthFiber
from qdc.qdc_result import QDCResult


def propagate_free_space(E, dz, wavelength, dx):
    """
    FFT-based free-space propagation by distance dz.
    * E: field (2D or flattened), shape (n, n).
    * dz: propagation distance
    * wavelength: in microns
    * dx: pixel size in x & y (assumed the same in both directions)
    Returns a flattened field (same shape as input).
    """
    if E.ndim == 1:
        n = int(np.sqrt(E.size))
        E = E.reshape([n, n])

    fa = np.fft.fft2(E)
    freq_x = np.fft.fftfreq(E.shape[1], d=dx)
    freq_y = np.fft.fftfreq(E.shape[0], d=dx)
    freq_Xs, freq_Ys = np.meshgrid(freq_x, freq_y)

    light_k = 2 * np.pi / wavelength
    k_x = freq_Xs * 2 * np.pi
    k_y = freq_Ys * 2 * np.pi

    k_z_sqr = light_k**2 - (k_x**2 + k_y**2)
    # clamp negative => evanescent
    np.maximum(k_z_sqr, 0, out=k_z_sqr)
    k_z = np.sqrt(k_z_sqr)

    # free-space phase shift
    fa *= np.exp(1j * k_z * dz)

    out_E = np.fft.ifft2(fa)
    return out_E.ravel()


class QDCExperiment(object):
    def __init__(self, mw_fiber: ManyWavelengthFiber):
        """
        Orchestrates 'classical' and 'two-photon' style correlations
        using a ManyWavelengthFiber object that has a list of Fibers.
        """
        self.mwf = mw_fiber

    def get_classical_PCCs(self):
        i_ref = 0
        f_ref = self.mwf.fibers[i_ref]
        E_end0 = f_ref.propagate(show=False)
        I_end0 = np.abs(E_end0) ** 2
        # some cropping
        n = f_ref.npoints
        II0 = I_end0.reshape([n, n])[50:80, 50:80]

        pccs = np.zeros(len(self.mwf.fibers))
        for i, f in enumerate(self.mwf.fibers):
            E_end = f.propagate(show=False)
            I_end = np.abs(E_end) ** 2
            II = I_end.reshape([n, n])[50:80, 50:80]
            pccs[i] = np.corrcoef(II0.ravel(), II.ravel())[0, 1]

        delta_lambdas = np.array([f.wl - f_ref.wl for f in self.mwf.fibers])
        return delta_lambdas, pccs

    def get_klyshko_PCCs(self, dz=0):
        i_middle = len(self.mwf.fibers) // 2
        # number of measurements: degenerate + pairs
        N_measurements = (len(self.mwf.fibers) // 2) + 1
        pccs = np.zeros(N_measurements)
        delta_lambdas = np.zeros(N_measurements)

        # 1) degenerate => simply propagate fiber i_middle
        f_mid = self.mwf.fibers[i_middle]
        f_mid.set_input_gaussian(sigma=10, X0=3, Y0=9, X_linphase=0.3, random_phase=0.5)

        # first half fiber
        E_end0 = f_mid.propagate(show=False)

        # Free-space (dz)
        E_after_prop = propagate_free_space(
            E_end0, dz, f_mid.wl, f_mid.index_profile.dh
        )
        # Then again feed it to the same fiber
        f_mid.profile_0 = E_after_prop
        E_end0 = f_mid.propagate(show=False)
        I_end0 = np.abs(E_end0) ** 2
        n = f_mid.npoints
        II0 = I_end0.reshape([n, n])[50:80, 50:80]
        pccs[0] = 1.0
        delta_lambdas[0] = 0.0

        for di in range(1, N_measurements):
            f_plus = self.mwf.fibers[i_middle + di]
            f_minus = self.mwf.fibers[i_middle - di]

            # first half on f_plus
            f_plus.set_input_gaussian(sigma=10, X0=3, Y0=9, X_linphase=0.3, random_phase=0.5)
            E_end_plus = f_plus.propagate(show=False)

            # free space
            E_mid = propagate_free_space(E_end_plus, dz, f_plus.wl, f_plus.index_profile.dh)
            # second free space for minus
            E_mid = propagate_free_space(E_mid, dz, f_minus.wl, f_minus.index_profile.dh)

            # second half on f_minus
            f_minus.profile_0 = E_mid
            E_end_minus = f_minus.propagate(show=False)

            I_end = np.abs(E_end_minus) ** 2
            II = I_end.reshape([n, n])[50:80, 50:80]
            pccs[di] = np.corrcoef(II0.ravel(), II.ravel())[0, 1]
            delta_lambdas[di] = f_plus.wl - f_minus.wl

        return delta_lambdas, pccs

    def get_classical_PCCs_average(self, N_configs=5):
        pccs_all = np.zeros((N_configs, len(self.mwf.fibers)))
        delta_lambdas = None
        for i in tqdm(range(N_configs)):
            self.mwf.set_inputs_gaussian()
            dl, pccs = self.get_classical_PCCs()
            pccs_all[i, :] = pccs
            if delta_lambdas is None:
                delta_lambdas = dl
        return delta_lambdas, pccs_all.mean(axis=0)

    def get_klyshko_PCCs_average(self, N_configs=5, dz=0):
        pccs_all = []
        delta_lambdas = None
        for i in tqdm(range(N_configs)):
            dl, pcc = self.get_klyshko_PCCs(dz=dz)
            pccs_all.append(pcc)
            if delta_lambdas is None:
                delta_lambdas = dl
        pccs_mean = np.mean(np.array(pccs_all), axis=0)
        return delta_lambdas, pccs_mean

    def run_PCCs_different_dz(self, dzs=(0, 20, 40, 60, 80), N_classical=5, N_klyshko=2):
        result = QDCResult()
        print(f"Getting classical with average on {N_classical} ...")
        dl_classical, pcc_classical = self.get_classical_PCCs_average(N_classical)
        result.delta_lambdas_classical = dl_classical
        result.pccs_classical = pcc_classical

        for dz in dzs:
            print(f"Getting Klyshko with average on {N_klyshko}, dz={dz} ...")
            dl_k, pcc_k = self.get_klyshko_PCCs_average(N_klyshko, dz=dz)
            result.klyshko_by_dz[dz] = (dl_k, pcc_k)

        result.metadata["fiber_length"] = self.mwf.fibers[0].L
        result.metadata["dzs"] = dzs
        return result
