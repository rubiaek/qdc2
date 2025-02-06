import copy
import numpy as np
import matplotlib.pyplot as plt

class QDCResult(object):
    def __init__(self):
        # For “classical” correlation
        self.delta_lambdas_classical = None
        self.pccs_classical = None

        # For “Klyshko” or two-photon correlation vs. various dz
        # We'll store them in dicts so that a single QDCResult
        # can hold multiple runs at different dz values:
        self.klyshko_by_dz = {}  # dz -> (deltaLambda array, pcc array)

        # Possibly store any metadata
        self.metadata = {}

    def saveto(self, path):
        d = copy.deepcopy(self.__dict__)
        np.savez(path, **d)

    def loadfrom(self, path):
        data = np.load(path, allow_pickle=True)
        self.__dict__.update(data)

    def show(self, title="QDC Result", mode_mixing=0):
        """
        Plots classical data plus any two-photon data present.
        You can adapt to your exact labeling/needs.
        """
        fig, ax = plt.subplots()
        # Plot classical
        if self.delta_lambdas_classical is not None and self.pccs_classical is not None:
            ax.plot(
                self.delta_lambdas_classical * 1e3,
                self.pccs_classical,
                label="Classical",
                linewidth=3,
            )
        # Plot Klyshko for each dz
        for dz, (dl, pcc) in self.klyshko_by_dz.items():
            ax.plot(dl * 1e3, pcc, label=f"Klyshko dz={dz} μm")

        ax.set_xlabel(r"$\Delta \lambda$ (nm)")
        ax.set_ylabel("PCC")
        ax.legend()
        ax.set_title(f"{title}, mode mixing={mode_mixing}")
        plt.show()
