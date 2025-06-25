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


class QDCMMFExperiment(object):
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
        return delta_lambdas, pccs, classical_incoherent_sum.reshape([self.n, self.n])

    def get_SPDC_PCCs(self, dz=0):
        pccs = []
        delta_lambdas = []
        
        # Get reference from degenerate case
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
            
        # Average PCCs for each unique delta_lambda (since each delta_lambda appears twice)
        delta_lambdas = np.array(delta_lambdas)
        pccs = np.array(pccs)
        unique_dwl = np.unique(delta_lambdas)
        averaged_pccs = np.zeros_like(unique_dwl)
        for i, dwl in enumerate(unique_dwl):
            mask = delta_lambdas == dwl
            averaged_pccs[i] = np.mean(pccs[mask])
        
        return unique_dwl, averaged_pccs, SPDC_incoherent_sum.reshape([self.n, self.n])

    def get_PCCs_multi(self, mode='classical', N_configs=5, dz=0, g_params_list=None):
        pccs_all = []
        incoherent_sums = []
        
        for i in tqdm(range(N_configs), desc=f"Running {mode} measurements"):
            if g_params_list is not None:
                self.g_params = g_params_list[i]
            else:
                self.g_params = self.mwf.get_g_params(add_random=True)
            
            if mode == 'classical':
                dl, pccs, incoherent_sum = self.get_classical_PCCs()
                incoherent_sums.append(incoherent_sum)
            elif mode == 'SPDC':
                dl, pccs, incoherent_sum = self.get_SPDC_PCCs(dz=dz)
                incoherent_sums.append(incoherent_sum)
            else:
                raise ValueError("mode must be 'classical' or 'SPDC'")
                
            pccs_all.append(pccs)
            delta_lambdas = dl
                
        if mode == 'classical':
            self.result.delta_lambdas_classical = delta_lambdas
            self.result.classical_incoherent_sums = incoherent_sums
            self.result.classical_pccs_all = pccs_all
        elif mode == 'SPDC':
            self.result.SPDC_by_dz[dz] = (delta_lambdas, None)
            if not hasattr(self.result, 'SPDC_incoherent_sums_by_dz'):
                self.result.SPDC_incoherent_sums_by_dz = {}
            self.result.SPDC_incoherent_sums_by_dz[dz] = incoherent_sums
            if not hasattr(self.result, 'SPDC_pccs_all_by_dz'):
                self.result.SPDC_pccs_all_by_dz = {}
            self.result.SPDC_pccs_all_by_dz[dz] = pccs_all
        else:
            raise ValueError("mode must be 'classical' or 'SPDC'")

    def run_PCCs_different_dz(self, dzs=(0, 20, 40, 60, 80), N_classical=5, N_SPDC=2):
        # Each iteretion will have slightly different gaussian parameters
        g_params_list = [self.mwf.get_g_params(add_random=True) for _ in range(max(N_classical, N_SPDC))]

        print(f"Getting classical with average on {N_classical} ...")
        self.get_PCCs_multi(mode='classical', N_configs=N_classical, g_params_list=g_params_list)

        for dz in dzs:
            print(f"Getting SPDC with average on {N_SPDC}, dz={dz} ...")
            self.get_PCCs_multi(mode='SPDC', N_configs=N_SPDC, dz=dz, g_params_list=g_params_list)

        self.result.metadata["dzs"] = dzs
        self.result.metadata["N_classical"] = N_classical
        self.result.metadata["N_SPDC"] = N_SPDC
        self.result.metadata["gaussian_params"] = self.mwf.gaussian_params
        self.result.metadata["gaussian_dparams"] = self.mwf.gaussian_dparams
        self.result.metadata["g_params_list"] = g_params_list
        return self.result
