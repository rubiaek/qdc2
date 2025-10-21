import numpy as np
from tqdm import tqdm

from qdc.mmf.many_wl_fiber import ManyWavelengthFiber
from qdc.mmf.qdc_mmf_result import QDCMMFResult

def propagate_free_space(E, dz, wavelength, dx):
    """
    FFT-based free-space propagation by distance dz.
    * E: field (2D or flattened), shape (n, n).
    * dz: propagation distance
    * wavelength: in microns
    * dx: pixel size in x & y (assumed the same in both directions)
    Returns a flattened field (same shape as input).
    """
    if dz == 0:
        return E
    
    if E.ndim == 1:
        n = int(np.sqrt(E.size))
        E = E.reshape([n, n])

    # Embed in a larger array 
    n = E.shape[0]
    E_large = np.zeros((n*3, n*3), dtype=E.dtype)
    E_large[n:2*n, n:2*n] = E
    E = E_large

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

    # Return to fiber grid 
    out_E = out_E[n:2*n, n:2*n]
    return out_E.ravel()


class QDCMMFExperiment(object):
    def __init__(self, mw_fiber, free_mode_matrix=False):
        """
        Orchestrates 'classical' and 'two-photon' style correlations
        using a ManyWavelengthFiber object that has a list of Fibers.
        """
        self.mwf = mw_fiber
        self.result = QDCMMFResult()
        self._set_PCC_slice()
        self.g_params = self.mwf.get_g_params(add_random=True)
        self.n = self.mwf.fibers[0].npoints
        if not isinstance(self.n, int):
            self.n = self.n.item()
        self.free_mode_matrix = free_mode_matrix
        self.phase_matching_Lc = None  # in microns
        self.pump_waist = None  # in microns
        self.magnification = 5.0         # crystal -> fiber demagnification M
        self.wl_pump = 0.405             # pump λ (µm)
        self.n_pump = 1.692              # refractive index at pump λ
        self._pm_pump_amp = None
        self._pm_filter = None  
        self.excite_modes = None # Set to (start_mode, end_mode)
        self.random_mode_phases = None 
        self.input_slm_phases = False 
        self.slm_phases = None 

    def set_phase_matching(self, Lc_um, pump_waist_crystal, magnification=10, wl_pump=0.405, n_pump=1.692):
        self.phase_matching_Lc = Lc_um
        self.pump_waist_crystal = pump_waist_crystal
        self.magnification = magnification
        self.wl_pump = wl_pump
        self.n_pump = n_pump
        self.compute_phase_matching()

    def compute_phase_matching(self):
        self._pm_pump_amp = None
        self._pm_filter   = None
        n = self.n
        dx = self.mwf.dx
        M = float(self.magnification)

        # Real-space coordinates (centered)
        coords = (np.arange(n) - n//2) * dx
        X, Y = np.meshgrid(coords, coords)

        # Pump Gaussian at fiber plane (if any)
        if self.pump_waist_crystal is not None:
            w_fiber = self.pump_waist_crystal / M  # demagnified waist
            self._pm_pump_amp = np.exp(-(X**2 + Y**2) / (w_fiber**2))
        else:
            self._pm_pump_amp = None

        # k-space frequencies
        freq = np.fft.fftfreq(n, d=dx)            # cycles / µm
        q = 2 * np.pi * freq                      # rad / µm
        qx, qy = np.meshgrid(q, q)
        qsqr = (2*qx)**2 + (2*qy)**2  # factor of 2 assuming qs=-qi, so qs-qi=2qs

        # if x becomes smaller, qx becomes larger
        qsqrfiber = qsqr / (M**2)

        k_p = 2 * np.pi * self.n_pump / self.wl_pump
        arg = -(self.phase_matching_Lc / (4.0 * k_p)) * qsqrfiber  # -(Lc/(4k_p))|q_c|^2
        self._pm_filter = np.sinc(arg / np.pi)  # sinc(x) = sin(pi x)/(pi x)

    def _apply_phase_matching(self, E_flat):
        n = self.n
        E = E_flat.reshape(n, n)

        if self._pm_pump_amp is not None:
            E = E * self._pm_pump_amp

        if self._pm_filter is not None:
            F = np.fft.fft2(E)
            F *= self._pm_filter
            E_out = np.fft.ifft2(F)
        else:
            E_out = E

        return E_out.ravel()

    def _set_PCC_slice(self, n_pixels_diameter=30):
        self.result.metadata["PCC_slice_x"] = self.mwf.fibers[0].npoints//2 - n_pixels_diameter//2
        self.result.metadata["PCC_slice_y"] = self.mwf.fibers[0].npoints//2 - n_pixels_diameter//2
        self.result.metadata["PCC_slice_size"] = n_pixels_diameter

        self.PCC_slice = np.index_exp[self.result.metadata["PCC_slice_x"]:self.result.metadata["PCC_slice_x"] + self.result.metadata["PCC_slice_size"], self.result.metadata["PCC_slice_y"]:self.result.metadata["PCC_slice_y"] + self.result.metadata["PCC_slice_size"]]

    def set_input(self, fiber):
        if not self.input_slm_phases and self.excite_modes is None:
            fiber.set_input_gaussian(*self.g_params)
        elif self.input_slm_phases:
            fiber.set_input_gaussian(sigma=10)
            # Larger wavelength compared to design wavelength results with less phase accumulation 
            # (But this anyway has a pretty minor effect. even 0.7*phases results with a rather nice focus)
            SLM_phase_scale_factor = self.mwf.wl0 / fiber.wl  
            fiber.profile_0 = fiber.profile_0 * np.exp(1j * SLM_phase_scale_factor * self.slm_phases)
        elif self.excite_modes is not None:
            fiber.set_input_random_modes(self.excite_modes[0], self.excite_modes[1], self.random_mode_phases)

    def get_classical_PCCs(self, get_output_fields=False):
        i_ref = 0
        f_ref = self.mwf.fibers[i_ref]
        self.set_input(f_ref)
        E_end0 = f_ref.propagate(show=False, free_mode_matrix=False)
        I_end0 = np.abs(E_end0) ** 2

        # Gor a fair PCC, we need to normalize by the envelope, which is calculated by summing over all fiber modes        
        envelope = (np.abs(f_ref.modes.getModeMatrix())**2).sum(axis=1)
        envelope = envelope.reshape([self.n, self.n])[self.PCC_slice]

        # some cropping
        II0 = I_end0.reshape([self.n, self.n])[self.PCC_slice]  
        II0 = II0 / envelope

        classical_incoherent_sum = np.zeros_like(I_end0)

        pccs = np.zeros(len(self.mwf.fibers))
        output_fields = []
        for i, f in enumerate(self.mwf.fibers):
            self.set_input(f)
            E_end = f.propagate(show=False, free_mode_matrix=self.free_mode_matrix)
            I_end = np.abs(E_end) ** 2
            classical_incoherent_sum += I_end
            II = I_end.reshape([self.n, self.n])[self.PCC_slice]
            II = II / envelope
            pccs[i] = np.corrcoef(II0.ravel(), II.ravel())[0, 1]
            if get_output_fields:
                output_fields.append(E_end.reshape([self.n, self.n]))

        delta_lambdas = np.array([np.abs(f.wl - f_ref.wl) for f in self.mwf.fibers])
        if not get_output_fields:
            return delta_lambdas, pccs, classical_incoherent_sum.reshape([self.n, self.n])
        else:
            return delta_lambdas, pccs, classical_incoherent_sum.reshape([self.n, self.n]), np.array(output_fields)

    def get_SPDC_PCCs(self, dz=0, get_output_fields=False):
        pccs = []
        delta_lambdas = []
        output_fields = []
        
        # Get reference from degenerate case
        i_middle = len(self.mwf.fibers) // 2
        f_mid = self.mwf.fibers[i_middle]
        self.set_input(f_mid)
        E_end0 = f_mid.propagate(show=False, free_mode_matrix=False)

        envelope = (np.abs(f_mid.modes.getModeMatrix())**2).sum(axis=1)
        envelope = envelope.reshape([self.n, self.n])[self.PCC_slice]

        # Freespace back and forth 
        E_mid = propagate_free_space(E_end0, dz, f_mid.wl, self.mwf.dx)
        
        E_mid = self._apply_phase_matching(E_mid)
        
        E_mid = propagate_free_space(E_mid, dz, f_mid.wl, self.mwf.dx)
        # Then back into the same fiber
        f_mid.profile_0 = E_mid
        E_end0 = f_mid.propagate(show=False, free_mode_matrix=self.free_mode_matrix)
        I_end0 = np.abs(E_end0) ** 2

        II0 = I_end0.reshape([self.n, self.n])[self.PCC_slice]
        II0 = II0 / envelope
                
        SPDC_incoherent_sum = I_end0.copy()

        # Iterate through all wavelengths and pair each with its opposite
        for i, f_plus in enumerate(self.mwf.fibers):
            f_minus = self.mwf.fibers[len(self.mwf.fibers) - i - 1]
            
            # first half on f_plus
            self.set_input(f_plus)
            E_end_plus = f_plus.propagate(show=False, free_mode_matrix=self.free_mode_matrix)

            # free space, each time with the plus/minus wavelength
            E_mid = propagate_free_space(E_end_plus, dz, f_plus.wl, self.mwf.dx)
            
            E_mid = self._apply_phase_matching(E_mid)

            E_mid = propagate_free_space(E_mid, dz, f_minus.wl, self.mwf.dx)

            # second half on f_minus
            f_minus.profile_0 = E_mid
            E_end_minus = f_minus.propagate(show=False, free_mode_matrix=self.free_mode_matrix)
            if get_output_fields:
                output_fields.append(E_end_minus)

            I_end = np.abs(E_end_minus) ** 2
            SPDC_incoherent_sum += I_end
            II = I_end.reshape([self.n, self.n])[self.PCC_slice]
            II = II / envelope
            pccs.append(np.corrcoef(II0.ravel(), II.ravel())[0, 1])
            delta_lambdas.append(np.abs(f_plus.wl - f_minus.wl))
            
        # Average PCCs for each unique delta_lambda (since each delta_lambda appears twice)
        delta_lambdas = np.array(delta_lambdas)
        pccs = np.array(pccs)
        unique_dwl = np.unique(delta_lambdas)
        averaged_pccs = np.zeros_like(unique_dwl)
        if get_output_fields:
            output_fields = np.array(output_fields)
            unique_output_fields = []
        for i, dwl in enumerate(unique_dwl):
            mask = delta_lambdas == dwl
            averaged_pccs[i] = np.mean(pccs[mask])

            if get_output_fields:
                relevant_outputs = output_fields[mask]
                unique_output_fields.append(relevant_outputs[0])

        if get_output_fields:
            return unique_dwl, averaged_pccs, SPDC_incoherent_sum.reshape([self.n, self.n]), np.array(unique_output_fields)
        else:
            return unique_dwl, averaged_pccs, SPDC_incoherent_sum.reshape([self.n, self.n])

    def get_PCCs_multi(self, mode='classical', N_configs=5, dz=0, g_params_list=None):
        pccs_all = []
        incoherent_sums = []
        
        for i in tqdm(range(N_configs), desc=f"Running {mode} measurements"):
            if g_params_list is not None:
                self.g_params = g_params_list[i]
            else:
                self.g_params = self.mwf.get_g_params(add_random=True)
            
            if self.excite_modes is not None:
                # self.random_mode_phases = np.exp(1j * self.mwf.rng.uniform(0, 2*np.pi, self.excite_modes[1] - self.excite_modes[0]))
                # Flat phases for now 
                self.random_mode_phases = np.exp(1j * 0 * self.mwf.rng.uniform(0, 2*np.pi, self.excite_modes[1] - self.excite_modes[0]))

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
            self.result.SPDC_incoherent_sums_by_dz[dz] = incoherent_sums
            if not hasattr(self.result, 'SPDC_pccs_all_by_dz'):
                self.result.SPDC_pccs_all_by_dz = {}
            self.result.SPDC_pccs_all_by_dz[dz] = pccs_all
        else:
            raise ValueError("mode must be 'classical' or 'SPDC'")

    def run_PCCs_different_dz(self, dzs=(0, 20, 40, 60, 80), N_classical=5, N_SPDC=2, g_params_list=None):
        # Each iteretion will have slightly different gaussian parameters
        if g_params_list is None:
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
