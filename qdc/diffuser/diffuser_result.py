import numpy as np
import copy

class DiffuserResult:
    def __init__(self):
        self.intensity_map = None
        self.wavelengths = None

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

    def saveto(self, path):
        d = copy.deepcopy(self.__dict__)
        np.savez(path, **d)

    def loadfrom(self, path):
        data = np.load(path, allow_pickle=True)
        self.__dict__.update(data)
