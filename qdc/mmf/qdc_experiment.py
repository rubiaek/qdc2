import numpy as np
from tqdm import tqdm

from qdc.mmf.many_wl_fiber import ManyWavelengthFiber
from qdc.mmf.qdc_result import QDCMMFResult

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
        self.result = QDCMMFResult()
        self.result.metadata["PCC_slice_x"] = self.mwf.fibers[0].npoints//2 - 15
        self.result.metadata["PCC_slice_y"] = self.mwf.fibers[0].npoints//2 - 15
        self.result.metadata["PCC_slice_size"] = 30
        self.PCC_slice = np.index_exp[self.result.metadata["PCC_slice_x"]:self.result.metadata["PCC_slice_x"] + self.result.metadata["PCC_slice_size"], self.result.metadata["PCC_slice_y"]:self.result.metadata["PCC_slice_y"] + self.result.metadata["PCC_slice_size"]]
        self.g_params = self.mwf.get_g_params(add_random=True)
        self.n = self.mwf.fibers[0].npoints

    def get_classical_PCCs(self):
        i_ref = 0
        f_ref = self.mwf.fibers[i_ref]
        f_ref.set_input_gaussian(*self.g_params)
        E_end0 = f_ref.propagate(show=False)
        I_end0 = np.abs(E_end0) ** 2
        # some cropping
        II0 = I_end0.reshape([self.n, self.n])[self.PCC_slice]  

        classical_incoherent_sum = np.zeros_like(I_end0)

        pccs = np.zeros(len(self.mwf.fibers))
        for i, f in enumerate(self.mwf.fibers):
            f.set_input_gaussian(*self.g_params)
            E_end = f.propagate(show=False)
            I_end = np.abs(E_end) ** 2
            classical_incoherent_sum += I_end
            II = I_end.reshape([self.n, self.n])[self.PCC_slice]
            pccs[i] = np.corrcoef(II0.ravel(), II.ravel())[0, 1]

        delta_lambdas = np.array([np.abs(f.wl - f_ref.wl) for f in self.mwf.fibers])
        self.result.classical_incoherent_sum = classical_incoherent_sum.reshape([self.n, self.n])
        return delta_lambdas, pccs

    def get_klyshko_PCCs(self, dz=0, add_random=True):
        pccs = []
        delta_lambdas = []
        
        # Get reference field for correlation (using middle fiber)
        i_middle = len(self.mwf.fibers) // 2
        f_mid = self.mwf.fibers[i_middle]
        f_mid.set_input_gaussian(*self.g_params)
        E_end0 = f_mid.propagate(show=False)

        # Freespace back and forth 
        E_after_prop = propagate_free_space(E_end0, 2*dz, f_mid.wl, self.mwf.dx)
        # Then back into the same fiber
        f_mid.profile_0 = E_after_prop
        E_end0 = f_mid.propagate(show=False)
        I_end0 = np.abs(E_end0) ** 2

        II0 = I_end0.reshape([self.n, self.n])[self.PCC_slice]
        
        SPDC_incoherent_sum = I_end0.copy()

        # Iterate through all wavelengths and pair each with its opposite
        for i, f_plus in enumerate(self.mwf.fibers):
            f_minus = self.mwf.fibers[len(self.mwf.fibers) - i - 1]
            
            # first half on f_plus
            f_plus.set_input_gaussian(*self.g_params)
            E_end_plus = f_plus.propagate(show=False)

            # free space, each time with the plus/minus wavelength
            E_mid = propagate_free_space(E_end_plus, dz, f_plus.wl, self.mwf.dx)
            E_mid = propagate_free_space(E_mid, dz, f_minus.wl, self.mwf.dx)

            # second half on f_minus
            f_minus.profile_0 = E_mid
            E_end_minus = f_minus.propagate(show=False)

            I_end = np.abs(E_end_minus) ** 2
            SPDC_incoherent_sum += I_end
            II = I_end.reshape([self.n, self.n])[self.PCC_slice]
            pccs.append(np.corrcoef(II0.ravel(), II.ravel())[0, 1])
            delta_lambdas.append(np.abs(f_plus.wl - f_minus.wl))
            
        self.result.SPDC_incoherent_sum = SPDC_incoherent_sum.reshape([self.n, self.n])

        return np.array(delta_lambdas), np.array(pccs)

    # TODO: combine both of these to one function, it will also make sure the same g_params are used for classical and SPDC simulations
    def get_classical_PCCs_average(self, N_configs=5):
        pccs_all = np.zeros((N_configs, len(self.mwf.fibers)))
        delta_lambdas = None
        for i in tqdm(range(N_configs)):
            self.g_params = self.mwf.get_g_params(add_random=True)
            dl, pccs = self.get_classical_PCCs()
            pccs_all[i, :] = pccs
            if delta_lambdas is None:
                delta_lambdas = dl
        return delta_lambdas, pccs_all.mean(axis=0)

    def get_klyshko_PCCs_average(self, N_configs=5, dz=0):
        pccs_all = []
        delta_lambdas = None
        for i in tqdm(range(N_configs)):
            self.g_params = self.mwf.get_g_params(add_random=True)
            dl, pcc = self.get_klyshko_PCCs(dz=dz)
            pccs_all.append(pcc)
            if delta_lambdas is None:
                delta_lambdas = dl
        pccs_mean = np.mean(np.array(pccs_all), axis=0)
        return delta_lambdas, pccs_mean

    def run_PCCs_different_dz(self, dzs=(0, 20, 40, 60, 80), N_classical=5, N_klyshko=2):
        """
        Returns a QDCResult containing classical and klyshko data for multiple dz.
        """

        # classical
        print(f"Getting classical with average on {N_classical} ...")
        dl_classical, pcc_classical = self.get_classical_PCCs_average(N_classical)
        self.result.delta_lambdas_classical = dl_classical
        self.result.pccs_classical = pcc_classical

        # klyshko
        for dz in dzs:
            print(f"Getting Klyshko with average on {N_klyshko}, dz={dz} ...")
            dl_k, pcc_k = self.get_klyshko_PCCs_average(N_klyshko, dz=dz)
            self.result.klyshko_by_dz[dz] = (dl_k, pcc_k)

        self.result.metadata["dzs"] = dzs
        self.result.metadata["N_classical"] = N_classical
        self.result.metadata["N_klyshko"] = N_klyshko
        self.result.metadata["gaussian_params"] = self.mwf.gaussian_params
        self.result.metadata["gaussian_dparams"] = self.mwf.gaussian_dparams
        return self.result
