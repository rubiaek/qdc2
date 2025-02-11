from pywin.scintilla.scintillacon import SCE_PS_DSC_COMMENT

from qdc.diffuser.field import Field
import numpy as np
import matplotlib.pyplot as plt
import copy


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


    def _populate_res_SPDC(self):
        self.SPDC_fields = []
        for field_E, wl in zip(self._SPDC_fields_E, self._SPDC_fields_wl):
            self.SPDC_fields.append(Field(self.x, self.y, wl, field_E))

        self.SPDC_fields = np.array(self.SPDC_fields)

        PCCs = []
        f0 = self.SPDC_fields[0]
        for f in self.SPDC_fields:
            PCC = np.corrcoef(f0.I.ravel(), f.I.ravel())[0, 1]
            PCCs.append(PCC)
        self.SPDC_PCCs = np.array(PCCs)

        final_I = np.zeros_like(self.SPDC_fields[0].I)
        for field in self.SPDC_fields:
            final_I += field.I
        self.SPDC_incoherent_sum = final_I

    def _populate_res_classical(self):
        self.classical_fields = []
        for field_E, wl in zip(self._classical_fields_E, self._classical_fields_wl):
            self.classical_fields.append(Field(self.x, self.y, wl, field_E))

        self.classical_fields = np.array(self.classical_fields)

        PCCs = []
        f0 = self.classical_fields[0]
        for f in self.classical_fields:
            PCC = np.corrcoef(f0.I.ravel(), f.I.ravel())[0, 1]
            PCCs.append(PCC)
        self.classical_PCCs = np.array(PCCs)

        final_I = np.zeros_like(self.classical_fields[0].I)
        for field in self.classical_fields:
            final_I += field.I
        self.classical_incoherent_sum = final_I

    # contrast =  np.std(data) / np.mean(data)

    def plot_PCCs_SPDC(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.SPDC_delta_lambdas*1e9, self.SPDC_PCCs, '*--', label='PCC')
        ax.set_xlabel('$\Delta\lambda$ [nm]')
        ax.set_ylabel('PCC')
        ax.set_title('SPDC experiment')
        ax.figure.show()

    def show_incoherent_sum_SPDC(self, ax=None):
        Field(self.x, self.y, self.wavelengths[0], np.sqrt(self.SPDC_incoherent_sum)).show(title='Incoherent sum of all wavelengths SPDC', ax=ax)

    def plot_PCCs_classical(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.classical_delta_lambdas*1e9, self.classical_PCCs, '*--', label='PCC')
        ax.set_xlabel('$\Delta\lambda$ [nm]')
        ax.set_ylabel('PCC')
        ax.set_title('Classical experiment')
        ax.figure.show()

    def show_incoherent_sum_classical(self, ax=None):
        Field(self.x, self.y, self.wavelengths[0], np.sqrt(self.classical_incoherent_sum)).show(title='Incoherent sum of all wavelengths classical', ax=ax)

    def show(self):
        fig, axes = plt.subplots(2, 2, figsize=(11,10))
        self.show_incoherent_sum_SPDC(axes[0, 0])
        self.show_incoherent_sum_classical(axes[0, 1])
        self.plot_PCCs_SPDC(axes[1, 0])
        self.plot_PCCs_classical(axes[1, 1])

    def show_diffuser(self):
        fig, ax = plt.subplots()
        pcm = ax.imshow(self.diffuser_mask, extent=[self.x[0] * 1e3, self.x[-1] * 1e3, self.y[0] * 1e3, self.y[-1] * 1e3],
                        cmap='viridis', origin='lower')
        fig.colorbar(pcm, ax=ax, label='Phase [rad]')
        ax.set_title("Single Diffuser Phase")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.show()

    def saveto(self, path, save_fields=False):
        d = copy.deepcopy(self.__dict__)
        d.pop('SPDC_fields')
        if not save_fields:
            d.pop('_SPDC_fields_E')
        np.savez(path, **d)

    def loadfrom(self, path):
        data = np.load(path, allow_pickle=True)
        self.__dict__.update(data)
