import copy
import numpy as np
import matplotlib.pyplot as plt

class QDCMMFResult(object):
    def __init__(self):
        # For “classical” correlation
        self.delta_lambdas_classical = None
        self.pccs_classical = None

        # For “Klyshko” or two-photon correlation vs. various dz
        # We'll store them in dicts so that a single QDCResult
        # can hold multiple runs at different dz values:
        self.klyshko_by_dz = {}  # dz -> (deltaLambda array, pcc array)

        self.classical_incoherent_sum = None         # 2D array
        self.klyshko_incoherent_sum_by_dz = {}       # dz -> 2D array

        # Possibly store any metadata
        self.metadata = {}

    def saveto(self, path):
        d = copy.deepcopy(self.__dict__)
        np.savez(path, **d)

    def loadfrom(self, path):
        data = np.load(path, allow_pickle=True)
        self.__dict__.update(data)

    def show(self, title='', saveto_path=''):
        """
        Plots classical data plus any two-photon data present.
        You can adapt to your exact labeling/needs.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot classical
        if self.delta_lambdas_classical is not None and self.pccs_classical is not None:
            ax.plot(
                self.delta_lambdas_classical * 1e3,
                self.pccs_classical,
                label="Classical",
                linewidth=3,
            )
        default_label = None
        linewidth = 1
        if len(self.klyshko_by_dz) == 1:
            default_label = 'SPDC'
            linewidth = 3

        # Plot Klyshko for each dz
        for dz, (dl, pcc) in self.klyshko_by_dz.items():
            ax.plot(dl * 1e3, pcc, label=default_label or f"Klyshko dz={dz} μm", linewidth=linewidth)

        ax.set_xlabel(r"$\Delta \lambda$ (nm)")
        ax.set_ylabel("PCC")
        ax.legend(fontsize=14)
        ax.set_title(f"{title}")
        plt.show()
        if saveto_path:
            fig.savefig(f"{saveto_path}.png")

    def show_incoherent_sum(self):
        fig, axes = plt.subplots(2, 1, figsize=(5, 10))
        imm = axes[0].imshow(self.classical_incoherent_sum)
        axes[0].set_title('Classical')
        fig.colorbar(imm, ax=axes[0])
        imm = axes[1].imshow(self.SPDC_incoherent_sum)
        axes[1].set_title('SPDC')
        fig.colorbar(imm, ax=axes[1])
        fig.show()
