import numpy as np
import copy
from qdc.diffuser.utils import prop_farfield_fft, phase_screen_diff, propagate_free_space
from qdc.diffuser.field import Field
from qdc.diffuser.diffuser_result import DiffuserResult


class DiffuserSimulation:
    def __init__(self,
                 Nx=512, Ny=512, Lx=2e-3, Ly=2e-3,
                 wl0=808e-9, Dwl=40e-9, N_wl=41,
                 waist=20e-6, distance_to_det=100e-3,
                 diffuser_angle=0.5):

        self.res = DiffuserResult()
        self.res.Nx = Nx
        self.res.Ny = Ny
        self.res.x = np.linspace(-Lx/2, Lx/2, Nx)
        self.res.y = np.linspace(-Ly/2, Ly/2, Ny)
        self.res.wl0 = wl0
        self.res.Dwl = Dwl
        self.res.N_wl = N_wl
        self.res.waist = waist
        self.res.dz = distance_to_det
        self.res.diffuser_angle = diffuser_angle
        self.res.wavelengths = np.linspace(self.wl_center - self.wl_half_range,
                              self.wl_center + self.wl_half_range,
                              self.Nwl)


    def __getattr__(self, name):
        # This method is only called if 'name' is not found in the instance
        return getattr(self.res, name)


    def _get_wl_range(self):
        """
        Return array of N_wl wavelengths spanning +/- Dwl/2 around wl0.
        """
        c = 299792458e6  # speed of light in um/s
        f0 = c / self.wl0
        frange = (c / self.wl0**2) * self.Dwl
        df = frange / self.N_wl
        f = f0 + np.arange(-self.N_wl / 2, self.N_wl / 2) * df
        l = c / f
        return l[::-1]

    def make_detection_gaussian(self, lam):
        """
        Make a Gaussian beam at the detection plane with self.waist.
        """
        XX, YY = np.meshgrid(self.x, self.y)
        r2 = XX**2 + YY**2
        E = np.exp(-r2 / (self.waist**2), dtype=np.complex128)
        E /= np.sqrt((np.abs(E)**2).sum())
        return Field(self.x, self.y, lam, E)

    def run_SPDC_simulation(self) -> (list[float], list[Field]):
        """ returns list of output fields and one-sided delta lambdas"""
        i_middle = len(self.N_wl) // 2
        # number of measurements: degenerate + pairs
        N_measurements = (self.N_wl // 2) + 1
        fields = []
        delta_lambdas = []

        diffuser_mask = phase_screen_diff(self.x, self.y, self.wl_center, self.diffuser_angle)

        # degenerate forward pass
        field_det = self.make_detection_gaussian(self.wl_center)
        field_crystal = propagate_free_space(field_det, self.dz)
        field_crystal.E *= np.exp(1j * diffuser_mask)
        # here "switch wl", but degenerate
        field_crystal.E *= np.exp(1j * diffuser_mask)
        field_det_new = propagate_free_space(field_det, self.dz)

        delta_lambdas.append(0.0)
        fields.append(field_det_new)

        # non-degenerate
        for di in range(1, N_measurements):
            wl_plus = self.wavelengths[i_middle + di]
            wl_minus = self.wavelengths[i_middle - di]

            field_det = self.make_detection_gaussian(wl_plus)
            field_crystal = propagate_free_space(field_det, self.dz)
            field_crystal.E *= np.exp(1j * diffuser_mask*self.wl0/wl_plus)
            field_after_crystal = copy.deepcopy(field_crystal)
            field_after_crystal.wl = wl_minus
            field_after_crystal.E *= np.exp(1j * diffuser_mask*self.wl0/wl_minus)
            field_det_new = propagate_free_space(field_det, self.dz)

            delta_lambdas.append(wl_plus - wl_minus)
            fields.append(field_det_new)

        return delta_lambdas, fields
