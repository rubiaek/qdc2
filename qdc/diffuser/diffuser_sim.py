import numpy as np
import copy

from qdc.diffuser.utils import prop_farfield_fft, backprop_farfield_fft, ft2, ift2, phase_screen_diff


class Field:
    """
    Simple container for a 2D field:
      - E: 2D complex ndarray
      - x, y: 1D coordinate arrays
      - lam: wavelength in meters
      - k: wave number = 2*pi/lam
    """
    def __init__(self, x, y, lam, E=None):
        self.x = x
        self.y = y
        self.lam = lam
        self.k = 2.0*np.pi / lam
        if E is None:
            Nx = len(x)
            Ny = len(y)
            self.E = np.zeros((Ny, Nx), dtype=np.complex128)
        else:
            self.E = E

class DiffuserSimulation:
    def __init__(self,
                 Nx=512, Ny=512,
                 Lx=2e-3, Ly=2e-3,
                 lam_center=808e-9,
                 lam_half_range=40e-9,
                 Nwl=5,
                 waist=20e-6,
                 focal_length=100e-3,
                 diffuser_angle=0.5):
        self.Nx = Nx
        self.Ny = Ny
        self.x = np.linspace(-Lx/2, Lx/2, Nx)
        self.y = np.linspace(-Ly/2, Ly/2, Ny)
        self.lam_center = lam_center
        self.lam_half_range = lam_half_range
        self.Nwl = Nwl
        self.waist = waist
        self.f = focal_length
        self.diffuser_angle = diffuser_angle

    def make_detection_gaussian(self, lam):
        """
        Make a Gaussian beam at the detection plane with waist = self.waist.
        """
        XX, YY = np.meshgrid(self.x, self.y, indexing='xy')
        r2 = XX**2 + YY**2
        E = np.exp(-r2 / (self.waist**2))
        return Field(self.x, self.y, lam, E)

    def run_simulation(self):
        """
        1) Generate a detection-mode Gaussian at the center wavelength
        2) Back-propagate (Klyshko picture) to the crystal/diffuser plane
        3) For each wavelength in [lam_center - lam_half_range, lam_center + lam_half_range],
           a) Create a random phase screen for that lam
           b) Multiply the field at crystal plane
           c) Forward-propagate to detection
           d) Incoherently sum intensities
        Return final summed intensity map.
        """
        lam_vec = np.linspace(self.lam_center - self.lam_half_range,
                              self.lam_center + self.lam_half_range,
                              self.Nwl)

        # Start with detection field at lam_center
        field_det = self.make_detection_gaussian(self.lam_center)

        # Back-propagate to crystal plane
        field_crystal = backprop_farfield_fft(field_det, self.f)

        # We keep the amplitude from the previous step as the "base" field
        base_crystal_E = field_crystal.E

        # Prepare final intensity accumulator
        I_sum = np.zeros((self.Ny, self.Nx), dtype=np.float64)

        for lam in lam_vec:
            # Update the base_crystal field to the new wavelength
            # for proper forward propagation scale
            field_c = Field(self.x, self.y, lam, copy.deepcopy(base_crystal_E))

            # Generate random phase screen with wavelength dependence
            ph_screen = phase_screen_diff(field_c.x, field_c.y, lam, self.diffuser_angle)

            # Multiply field by e^{i * phase_screen}
            field_c.E *= np.exp(1j * ph_screen)

            # Forward-propagate from crystal to detection
            field_det_new = prop_farfield_fft(field_c, self.f)

            # Accumulate intensity (incoherent sum)
            I_sum += np.abs(field_det_new.E)**2

        # Average or keep total. Let's keep the average.
        I_sum /= self.Nwl
        return I_sum
