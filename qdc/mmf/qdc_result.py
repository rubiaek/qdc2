import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class QDCMMFResult(object):
    def __init__(self):
        # For “classical” correlation
        self.delta_lambdas_classical = None
        self.pccs_classical = None
        self.SPDC_by_dz = {}  # dz -> (deltaLambda array, pcc array)

        self.classical_incoherent_sum = None         # 2D array
        self.SPDC_incoherent_sum = None       # dz -> 2D array

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
        if len(self.SPDC_by_dz) == 1:
            default_label = 'SPDC'
            linewidth = 3

        for dz, (dl, pcc) in self.SPDC_by_dz.items():
            ax.plot(dl * 1e3, pcc, label=default_label or f"SPDC dz={dz} μm", linewidth=linewidth)

        ax.set_xlabel(r"$\Delta \lambda$ (nm)", fontsize=14)
        ax.set_ylabel("PCC", fontsize=14)
        ax.legend(fontsize=14)
        ax.set_title(f"{title}")
        fig.show()
        if saveto_path:
            fig.savefig(f"{saveto_path}.png")

    def show_incoherent_sum(self):
        fig, axes = plt.subplots(2, 1, figsize=(5, 8))
        imm = axes[0].imshow(self.classical_incoherent_sum)
        # axes[0].set_title('Classical')
        fig.colorbar(imm, ax=axes[0])
        square = patches.Rectangle(
            (self.metadata["PCC_slice_x"], self.metadata["PCC_slice_y"]),  # (x0, y0)
            self.metadata["PCC_slice_size"], self.metadata["PCC_slice_size"],  # width, height  
            linewidth=0.7,  # thin
            edgecolor='white',
            facecolor='none',
            linestyle = 'dashed'

        )
        axes[0].add_patch(square)
        axes[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        imm = axes[1].imshow(self.SPDC_incoherent_sum)
        # axes[1].set_title('SPDC')
        fig.colorbar(imm, ax=axes[1])
        square = patches.Rectangle(
            (self.metadata["PCC_slice_x"], self.metadata["PCC_slice_y"]),  # (x0, y0)
            self.metadata["PCC_slice_size"], self.metadata["PCC_slice_size"],  # width, height
            linewidth=0.7,  # thin
            edgecolor='white',
            facecolor='none',
            linestyle='dashed'
        )
        axes[1].add_patch(square)
        axes[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        fig.show()
