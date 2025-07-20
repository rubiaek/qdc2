from qdc.diffuser.field import Field
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import cv2
from matplotlib.widgets import Button


class DiffuserResult:
    def __init__(self):
        self._SPDC_fields_E = None
        self._SPDC_fields_wl = None
        self.SPDC_fields = None
        self.SPDC_delta_lambdas = None
        self._classical_fields_E = None
        self._classical_fields_wl = None
        self.classical_fields = None
        self.classical_delta_lambdas = None
        self.wavelengths = None
        self.SPDC_PCCs = None
        self.SPDC_incoherent_sum = None
        self.classical_PCCs = None
        self.classical_incoherent_sum = None
        self.D = None

        self.x = None
        self.y = None
        self.classical_ff_method = ''
        self.classical_xs = []
        self.classical_ys = []
        self.SPDC_ff_method = ''
        self.SPDC_xs = []
        self.SPDC_ys = []

    @property
    def dx(self):
        max_Xs = [x.max() for x in self.classical_xs]
        # Find the index of the field with the smallest x-range
        min_idx = np.argmin(max_Xs)
        global_x = self.classical_xs[min_idx]
        dx = global_x[1] - global_x[0]
        return dx

    def _get_global_grid(self, fields):
        max_Xs = [f.x.max() for f in fields]
        # Find the index of the field with the smallest x-range
        min_idx = np.argmin(max_Xs)
        global_x = fields[min_idx].x
        global_y = fields[min_idx].y
        return global_x, global_y

    def fix_grids(self, fields):
        from scipy.interpolate import RegularGridInterpolator
        new_fs = []
        global_x, global_y = self._get_global_grid(fields)
        global_YY, global_XX = np.meshgrid(global_y, global_x, indexing='ij')
        points = np.array([global_YY.flatten(), global_XX.flatten()]).T
        for f in fields:
            # The data in f.E is shaped (ny, nx) and RegularGridInterpolator expects
            # inputs in the order of the grid dimensions, which is (y, x)
            interp_func = RegularGridInterpolator((f.y, f.x), f.E)
            interpolated = interp_func(points).reshape(global_YY.shape)
            new_fs.append(Field(global_x, global_y, f.wl, interpolated))
        return new_fs

    def _populate_res_SPDC(self, D=15, fix_grids=True):
        mid = self.Nx // 2
        roi = np.index_exp[mid - D:mid + D, mid - D:mid + D]
        self.D = D
        self.SPDC_fields = []

        for field_E, wl, x, y in zip(self._SPDC_fields_E, self._SPDC_fields_wl,
                                        self.SPDC_xs, self.SPDC_ys):
            self.SPDC_fields.append(Field(x, y, wl, field_E))
        if fix_grids:
            self.SPDC_fields = self.fix_grids(self.SPDC_fields)
        # Even if didn't fix the grids, I still need to put there something...
        self.global_x_SPDC, self.global_y_SPDC = self.SPDC_fields[0].x, self.SPDC_fields[0].y

        self.SPDC_fields = np.array(self.SPDC_fields)

        PCCs = []
        degenerate_index = np.where(self.SPDC_delta_lambdas == 0)[0][0]
        f0 = self.SPDC_fields[degenerate_index]
        for f in self.SPDC_fields:
            PCC = np.corrcoef(f0.I[roi].ravel(), f.I[roi].ravel())[0, 1]
            PCCs.append(PCC)
        self.SPDC_PCCs = np.array(PCCs)

        unique_dwl = np.unique(self.SPDC_delta_lambdas)
        averaged_pccs = np.zeros_like(unique_dwl)
        for i, dwl in enumerate(unique_dwl):
            mask = self.SPDC_delta_lambdas == dwl
            averaged_pccs[i] = np.mean(self.SPDC_PCCs[mask])
        self.SPDC_delta_lambdas = unique_dwl
        self.SPDC_PCCs = averaged_pccs

        final_I = np.zeros_like(self.SPDC_fields[0].I)
        for field in self.SPDC_fields:
            final_I += field.I
        self.SPDC_incoherent_sum = final_I

    def _populate_res_classical(self, D=15, fix_grids=True):
        mid = self.Nx // 2
        roi = np.index_exp[mid - D:mid + D, mid - D:mid + D]
        self.D = D
        self.classical_fields = []

        for field_E, wl, x, y in zip(self._classical_fields_E, self._classical_fields_wl,
                                        self.classical_xs, self.classical_ys):
            self.classical_fields.append(Field(x, y, wl, field_E))
        if fix_grids:
            self.classical_fields = self.fix_grids(self.classical_fields)
        self.global_x_classical, self.global_y_classical = self.classical_fields[0].x, self.classical_fields[0].y

        self.classical_fields = np.array(self.classical_fields)

        PCCs = []
        f0 = self.classical_fields[0]
        for f in self.classical_fields:
            PCC = np.corrcoef(f0.I[roi].ravel(), f.I[roi].ravel())[0, 1]
            PCCs.append(PCC)
        self.classical_PCCs = np.array(PCCs)

        final_I = np.zeros_like(self.classical_fields[0].I)
        for field in self.classical_fields:
            final_I += field.I
        self.classical_incoherent_sum = final_I

    def plot_PCCs_SPDC(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.SPDC_delta_lambdas*1e9, self.SPDC_PCCs, '*--', label='PCC')
        ax.set_xlabel('$\Delta\lambda$ [nm]')
        ax.set_ylabel('PCC')
        ax.set_title('SPDC experiment')
        # ax.figure.show()

    def show_incoherent_sum_SPDC(self, ax=None, lognorm=False, clean=False, title='Incoherent sum of all wavelengths SPDC', add_square=True):
        fig, ax = Field(self.global_x_SPDC, self.global_y_SPDC, self.wavelengths[0], np.sqrt(self.SPDC_incoherent_sum)).show(
            title=title, ax=ax, lognorm=lognorm, clean=clean)
        D = self.D
        D = D*self.dx * 1e6  # This 1e6 is because of the silly Field.show() stretching impl.
        x_c = y_c = 0
        if add_square:
            rect = patches.Rectangle(
                (x_c - D, y_c - D),  # Bottom-left corner
                D*2, D*2,  # Width, Height
                linewidth=1.5,  # thin
                edgecolor='white',
                facecolor='none',
                linestyle='dashed'
            )
            ax.add_patch(rect)
        ax.set_xlim(x_c - 2000, x_c + 2000)
        ax.set_ylim(y_c - 2000, y_c + 2000)
        return fig, ax

    def plot_PCCs_classical(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.classical_delta_lambdas*1e9, self.classical_PCCs, '*--', label='PCC')
        ax.set_xlabel('$\Delta\lambda$ [nm]')
        ax.set_ylabel('PCC')
        ax.set_title('Classical experiment')
        # ax.figure.show()

    def show_incoherent_sum_classical(self, ax=None, lognorm=False, clean=False, title='Incoherent sum of all wavelengths classical', add_square=True):
        fig, ax = Field(self.global_x_classical, self.global_y_classical, self.wavelengths[0], np.sqrt(self.classical_incoherent_sum)).show(
            title=title, ax=ax, lognorm=lognorm, clean=clean)
        D = self.D
        D = D*self.dx * 1e6  # This 1e6 is because of the silly Field.show() stretching impl.
        x_c = y_c = 0
        if add_square:
            rect = patches.Rectangle(
                (x_c - D, y_c - D),  # Bottom-left corner
                D*2, D*2,  # Width, Height
                linewidth=1.5,  # thin
                edgecolor='white',
                facecolor='none',
                linestyle='dashed'
            )
            ax.add_patch(rect)
        ax.set_xlim(x_c - 2000, x_c + 2000)
        ax.set_ylim(y_c - 2000, y_c + 2000)
        return fig, ax

    def show_PCCs(self, ax=None):
        # Show classical and SPDC PCCs on the same plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.SPDC_delta_lambdas*1e9, self.SPDC_PCCs, '-', label='SPDC')
        ax.plot(self.classical_delta_lambdas*1e9, self.classical_PCCs, '-', label='Classical', color='#8c564b')
        ax.set_xlabel('$\Delta\lambda$ [nm]')
        ax.set_ylabel('PCC')
        ax.legend(loc='center right')
        fig.show()


    def show_diffuser(self):
        fig, ax = plt.subplots()
        pcm = ax.imshow(self.diffuser_mask, extent=[self.x[0] * 1e3, self.x[-1] * 1e3, self.y[0] * 1e3, self.y[-1] * 1e3],
                        cmap='viridis', origin='lower')
        fig.colorbar(pcm, ax=ax, label='Phase [rad]')
        ax.set_title("Single Diffuser Phase")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        # fig.show()

    def saveto(self, path, save_fields=False):
        d = copy.deepcopy(self.__dict__)
        d.pop('SPDC_fields')
        d.pop('classical_fields')
        if not save_fields:
            d.pop('_SPDC_fields_E')
            d.pop('_classical_fields_E')
        np.savez(path, **d)

    def loadfrom(self, path):
        data = np.load(path, allow_pickle=True)
        self.__dict__.update(data)
