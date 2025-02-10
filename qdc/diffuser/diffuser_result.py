import numpy as np

class DiffuserResult:
    def __init__(self, intensity_map=None, wavelengths=None):
        self.intensity_map = intensity_map  # 2D numpy array
        self.wavelengths = wavelengths      # 1D array of lambda values (optional)

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

    def __repr__(self):
        if self.intensity_map is not None:
            return (f"<SPDCResult shape={self.intensity_map.shape} "
                    f"min={self.intensity_map.min():.3g}, max={self.intensity_map.max():.3g}>")
        else:
            return "<SPDCResult empty>"
