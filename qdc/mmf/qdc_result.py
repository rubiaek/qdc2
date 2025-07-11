import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from qdc.misc import show

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

    def show(self, title='', saveto_path='', iter_no=None, show0=True):
        """
        If iter_no is None, shows the average; otherwise, shows the specific iteration.
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        # SPDC
        for dz_key, (dl, _) in self.SPDC_by_dz.items():
            y = None
            label = None
            if iter_no is None:
                y = self.SPDC_pccs_average(dz_key)
                if show0:
                    label = 'SPDC'
                else:
                    label = f"SPDC dz={dz_key} μm"
            elif hasattr(self, 'SPDC_pccs_all_by_dz') and dz_key in self.SPDC_pccs_all_by_dz and len(self.SPDC_pccs_all_by_dz[dz_key]) > iter_no:
                y = self.SPDC_pccs_all_by_dz[dz_key][iter_no]
                label = f"SPDC iter {iter_no}, dz={dz_key} μm"
            if y is not None:
                if show0 and dz_key != 0:
                    pass
                else:
                    # dl is originally in um, convert to nm
                    ax.plot(dl * 1e3, y, '-', label=label, linewidth=2)

        # Classical
        y = None
        label = None
        if iter_no is None:
            y = self.classical_pccs_average
            label = "Classical"
        elif hasattr(self, 'classical_pccs_all') and len(self.classical_pccs_all) > iter_no:
            y = self.classical_pccs_all[iter_no]
            label = f"Classical iter {iter_no}"
        if y is not None:
            ax.plot(self.delta_lambdas_classical * 1e3, y, '-', label=label, linewidth=2, color='#8c564b')


        ax.set_xlabel(r"$\Delta \lambda$ (nm)", fontsize=14)
        ax.set_ylabel("PCC", fontsize=14)
        ax.legend(fontsize=14,  bbox_to_anchor=(1, 0.5))
        
        if show0:
            ax.legend(loc='center right', bbox_to_anchor=(1, 0.5), fontsize=14)
        else:
            ax.legend(loc='lower center',
                bbox_to_anchor=(0.5, 1.02),
                ncol=3,  # spread out the entries horizontally
                frameon=False)

        ax.set_title(f"{title}")
        show(fig)
        if saveto_path:
            fig.savefig(f"{saveto_path}.png")

    def show_incoherent_sum(self, iter_no=None, dz=0):
        if iter_no is None:
            iter_no = 0

        fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), constrained_layout=True)
        
        # Classical
        if len(self.classical_incoherent_sums) > iter_no:
            imm = axes[0].imshow(self.classical_incoherent_sums[iter_no])
        else:
            raise ValueError(f"No classical data available for iteration {iter_no}")
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

        # SPDC
        if dz in self.SPDC_incoherent_sums_by_dz and len(self.SPDC_incoherent_sums_by_dz[dz]) > iter_no:
            imm = axes[1].imshow(self.SPDC_incoherent_sums_by_dz[dz][iter_no])
        else:
            raise ValueError(f"No SPDC data available for iteration {iter_no}")
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
        
        show(fig)

    @property
    def classical_pccs_average(self):
        if hasattr(self, 'classical_pccs_all') and len(self.classical_pccs_all) > 0:
            return np.mean(np.array(self.classical_pccs_all), axis=0)
        return None

    def SPDC_pccs_average(self, dz):
        if hasattr(self, 'SPDC_pccs_all_by_dz') and dz in self.SPDC_pccs_all_by_dz and len(self.SPDC_pccs_all_by_dz[dz]) > 0:
            return np.mean(np.array(self.SPDC_pccs_all_by_dz[dz]), axis=0)
        return None
