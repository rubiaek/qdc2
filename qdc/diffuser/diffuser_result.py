from qdc.diffuser.field import Field
import numpy as np
import matplotlib.pyplot as plt
import copy


class DiffuserResult:
    def __init__(self):
        self._SPDC_fields_E = None
        self._SPDC_fields_wl = None
        self.SPDC_fields = []
        self.SPDC_delta_lambdas = None
        self.wavelengths = None


    def _populate_fields(self):
        for field_E, wl in zip(self._SPDC_fields_E, self._SPDC_fields_wl):
            self.SPDC_fields.append(Field(self.x, self.y, wl, field_E))

    def compute_contrast(self, roi=None):
        """
        Contrast = std(I)/mean(I) over entire image or ROI.
        roi = (x0, x1, y0, y1)
        """
        if roi is not None:
            x0, x1, y0, y1 = roi
            data = self.intensity_map[y0:y1, x0:x1]
        else:
            data = self.intensity_map
        return np.std(data) / np.mean(data)

    def plot_SPDC_PCCs(self):
        fig, ax = plt.subplots()
        PCCs = []
        f0 = self.SPDC_fields[0]
        for f in self.SPDC_fields:
            PCC = np.corrcoef(f0.I.ravel(), f.I.ravel())[0, 1]
            PCCs.append(PCC)
        ax.plot(self.SPDC_delta_lambdas*1e9, PCCs, '*--', label='PCC')
        ax.set_xlabel('$\Delta\lambda$ [nm]')
        ax.set_ylabel('PCC')
        fig.show()

    def show_incoherent_sum_SPDC(self):
        final_I = np.zeros_like(self.SPDC_fields[0].I)
        for field in self.SPDC_fields:
            final_I += field.I
        Field(self.x, self.y, self.wavelengths[0], np.sqrt(final_I)).show(title='Incoherent sum of all wavelengths')

    def saveto(self, path):
        d = copy.deepcopy(self.__dict__)
        np.savez(path, **d)

    def loadfrom(self, path):
        data = np.load(path, allow_pickle=True)
        self.__dict__.update(data)
