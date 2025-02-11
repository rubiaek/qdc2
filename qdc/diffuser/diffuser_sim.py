import numpy as np
import copy
from qdc.diffuser.utils import prop_farfield_fft, phase_screen_diff, propagate_free_space
from qdc.diffuser.field import Field
from qdc.diffuser.diffuser_result import DiffuserResult


class DiffuserSimulation:
    def __init__(self,
                 Nx=512, Ny=512, Lx=2e-3, Ly=2e-3,
                 wl0=808e-9, Dwl=40e-9, N_wl=41,
                 waist=20e-6, focal_length=100e-3,
                 init_off_axis=200e-6, diffuser_angle=0.5, achromat_lens=True):

        self.res = DiffuserResult()
        self.res.Nx = Nx
        self.res.Ny = Ny
        self.res.x = np.linspace(-Lx/2, Lx/2, Nx)
        self.res.y = np.linspace(-Ly/2, Ly/2, Ny)
        self.XX, self.YY = np.meshgrid(self.x, self.y)
        self.res.wl0 = wl0
        self.res.Dwl = Dwl
        self.res.N_wl = N_wl
        self.res.waist = waist
        self.res.f = focal_length
        self.res.init_off_axis = init_off_axis
        self.res.diffuser_angle = diffuser_angle
        self.res.wavelengths = self._get_wl_range()
        self.res.diffuser_mask = phase_screen_diff(self.x, self.y, self.wl0, self.diffuser_angle)
        self.res.achromat_lens = achromat_lens


    def __getattr__(self, name):
        # This method is only called if 'name' is not found in the instance
        if hasattr(self.res, name):
            return getattr(self.res, name)
        else:
            raise AttributeError(name)


    def _get_wl_range(self):
        """ Return array of N_wl wavelengths spanning +/- Dwl/2 around wl0 equally spaced in frequency. """
        c = 299792458e6  # speed of light in um/s
        f0 = c / self.wl0
        frange = (c / self.wl0**2) * self.Dwl
        df = frange / self.N_wl
        f = f0 + np.arange(-self.N_wl / 2, self.N_wl / 2) * df
        l = c / f
        return l[::-1]

    def make_detection_gaussian(self, lam):
        """ Make a Gaussian beam at the detection plane with self.waist. """
        r2 = (self.XX-self.init_off_axis)**2 + self.YY**2
        E = np.exp(-r2 / (self.waist**2), dtype=np.complex128)
        E /= np.sqrt((np.abs(E)**2).sum())
        return Field(self.x, self.y, lam, E)


    def get_lens_mask(self, f, wl):
        k = 2 * np.pi / wl
        # important -i, assuming freespace is with +i
        mask = np.exp(-1j * (self.XX ** 2 + self.YY ** 2) * k / (2 * f))
        return mask


    def run_SPDC_simulation(self):
        """ returns list of output fields and one-sided delta lambdas"""
        i_middle = self.N_wl // 2
        # number of measurements: degenerate + pairs
        N_measurements = (self.N_wl // 2) + 1
        fields = []
        delta_lambdas = []

        wl0 = self.wavelengths[i_middle]

        # degenerate forward pass
        field_det = self.make_detection_gaussian(wl0)
        field_lens = propagate_free_space(field_det, self.f)
        field_lens.E *= self.get_lens_mask(self.f, wl0)
        field_crystal = propagate_free_space(field_lens, self.f)
        field_crystal.E *= np.exp(1j * self.diffuser_mask)
        # here "switch wl", but degenerate
        field_crystal.E *= np.exp(1j * self.diffuser_mask)
        field_lens2 = propagate_free_space(field_crystal, self.f)
        field_lens2.E *= self.get_lens_mask(self.f, wl0)
        field_det_new = propagate_free_space(field_lens2, self.f)

        delta_lambdas.append(0.0)
        fields.append(field_det_new)

        # non-degenerate
        for di in range(1, N_measurements):
            wl_plus = self.wavelengths[i_middle + di]
            wl_minus = self.wavelengths[i_middle - di]

            field_det = self.make_detection_gaussian(wl_plus)
            field_lens = propagate_free_space(field_det, self.f)
            field_lens.E *= self.get_lens_mask(self.f, wl_plus if self.achromat_lens else wl0)
            field_crystal = propagate_free_space(field_lens, self.f)
            # width of diffuser works differently for different wavelengths,
            # and assuming a very thin diffuser that does only 2pi for wl0
            field_crystal.E *= np.exp(1j * self.diffuser_mask*self.wl0/wl_plus)

            # change wl at crystal (could potentially add phase matching etc.)
            field_crystal.wl = wl_minus

            field_crystal.E *= np.exp(1j * self.diffuser_mask*self.wl0/wl_minus)
            field_lens2 = propagate_free_space(field_crystal, self.f)
            field_lens2.E *= self.get_lens_mask(self.f, wl_minus if self.achromat_lens else wl0)
            field_det_new = propagate_free_space(field_lens2, self.f)
            delta_lambdas.append(wl_plus - wl_minus)
            fields.append(field_det_new)

        self.res._SPDC_fields_E = np.array([f.E for f in fields])
        self.res._SPDC_fields_wl = np.array([f.wl for f in fields])
        self.res.SPDC_delta_lambdas = np.array(delta_lambdas)
        self.res._populate_fields()
        return self.res
