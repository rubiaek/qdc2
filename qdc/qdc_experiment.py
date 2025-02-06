import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from qdc.many_wl_fiber import ManyWavelengthFiber
from qdc.qdc_result import QDCResult

class QDCExperiment(object):
    def __init__(self, mw_fiber: ManyWavelengthFiber):
        """
        Orchestrates 'classical' and 'two-photon' style correlations
        using a ManyWavelengthFiber object that has a list of Fibers.
        """
        self.mwf = mw_fiber  # store reference

    def get_classical_PCCs(self):
        """
        Reproduces old .get_classical_PCCs() logic:
          - Take the 'middle' or first fiber as a reference
          - Compare correlation with all other fibers
        """
        # For demonstration, let's keep index=0 as reference
        # i_ref = len(self.fibers) // 2
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
        """
        Reproduces old .get_klyshko_PCCs() logic:
          - The fiber in the middle is used for degenerate
          - Then pairs at +/- offsets from the center
          - Possibly do free space propagation by dz in between, etc.
        """
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
        # free space
        E_after_prop = self.mwf.propagate_free_space(E_end0, dz, f_mid)
        # second half fiber
        f_mid.profile_0 = E_after_prop
        E_end0 = f_mid.propagate(show=False)
        I_end0 = np.abs(E_end0) ** 2
        n = f_mid.npoints
        II0 = I_end0.reshape([n, n])[50:80, 50:80]
        pccs[0] = 1.0  # self-correlation
        delta_lambdas[0] = 0.0

        # 2) Non-degenerate pairs
        for di in range(1, N_measurements):
            f_plus = self.mwf.fibers[i_middle + di]
            f_minus = self.mwf.fibers[i_middle - di]

            # first half fiber for f_plus
            f_plus.set_input_gaussian(
                sigma=10, X0=3, Y0=9, X_linphase=0.3, random_phase=0.5
            )
            E_end_plus = f_plus.propagate(show=False)
            # free space
            E_mid = self.mwf.propagate_free_space(E_end_plus, dz, f_plus)
            E_mid = self.mwf.propagate_free_space(E_mid, dz, f_minus)
            # second half fiber for f_minus
            f_minus.profile_0 = E_mid
            E_end_minus = f_minus.propagate(show=False)

            I_end = np.abs(E_end_minus) ** 2
            II = I_end.reshape([n, n])[50:80, 50:80]
            pccs[di] = np.corrcoef(II0.ravel(), II.ravel())[0, 1]
            delta_lambdas[di] = f_plus.wl - f_minus.wl

        return delta_lambdas, pccs

    def get_classical_PCCs_average(self, N_configs=5):
        """
        Reproduces .get_classical_PCCs_average()
        Just does multiple random or random-phase inputs
        and averages the PCC.
        """
        pccs_all = np.zeros((N_configs, len(self.mwf.fibers)))
        delta_lambdas = None
        for i in tqdm(range(N_configs)):
            # Example: set input to a Gaussian with random phase
            self.mwf.set_inputs_gaussian()
            dl, pccs = self.get_classical_PCCs()
            pccs_all[i, :] = pccs
            if delta_lambdas is None:
                delta_lambdas = dl
        return delta_lambdas, pccs_all.mean(axis=0)

    def get_klyshko_PCCs_average(self, N_configs=5, dz=0):
        """
        Reproduces .get_klyshko_PCCs_average()
        """
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
        """
        Reproduces old .run_PCCs_different_dz().
        - Compute classical once with multiple configs
        - Then Klyshko for multiple dz values
        - Return a QDCResult that can be saved or .show()ed
        """
        result = QDCResult()

        # 1) get classical
        print(f"Getting classical with average on {N_classical} ...")
        dl_classical, pcc_classical = self.get_classical_PCCs_average(N_classical)
        result.delta_lambdas_classical = dl_classical
        result.pccs_classical = pcc_classical

        # 2) get Klyshko for multiple dz
        for dz in dzs:
            print(f"Getting Klyshko with average on {N_klyshko}, dz={dz} ...")
            dl_k, pcc_k = self.get_klyshko_PCCs_average(N_klyshko, dz=dz)
            result.klyshko_by_dz[dz] = (dl_k, pcc_k)

        # Just store some metadata
        result.metadata["fiber_length"] = self.mwf.fibers[0].L
        result.metadata["dzs"] = dzs
        return result
