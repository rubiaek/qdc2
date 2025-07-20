import numpy as np
from qdc.diffuser.utils import prop_farfield_fft
from qdc.diffuser.field import Field
from qdc.diffuser.diffuser_result import DiffuserResult
import cv2
import pyfftw
pyfftw.interfaces.cache.enable()


def macro_pixels_phase(x, y, theta, rms_height):
    # adding 0.01 to not get m*2*pi-eps
    assert np.abs((rms_height+0.01) % (2*np.pi)) < 0.1, "rms_height must be an integer amount of 2*pi, to avoid a significant DC component"
    Dx = x[-1] - x[0]
    Nx = len(x)
    macro_pixel_size = 1e-6 / theta
    macro_pixels_N = int(Dx / macro_pixel_size)
    A = np.random.uniform(0, rms_height, (macro_pixels_N, macro_pixels_N))
    A2 = cv2.resize(A, (Nx, Nx), interpolation=cv2.INTER_NEAREST)
    return A2


class DiffuserSimulation:
    def __init__(self,
                 Nx=512, Ny=512, Lx=2e-3, Ly=2e-3,
                 wl0=808e-9, Dwl=40e-9, N_wl=41,
                 waist=20e-6, focal_length=100e-3,
                 init_off_axis=200e-6, diffuser_angle=0.5, rms_height=5, 
                 with_dispersion=True):

        self.res = DiffuserResult()
        self.res.Nx = Nx
        self.res.Ny = Ny
        self.res.x = np.linspace(-Lx/2, Lx/2, Nx)
        self.res.y = np.linspace(-Ly/2, Ly/2, Ny)
        self.XX, self.YY = np.meshgrid(self.x, self.y)
        self.res.wl0 = wl0
        self.res.Dwl = Dwl
        self.res.N_wl = N_wl
        assert N_wl % 2 == 1, "N_wl must be odd"
        self.res.waist = waist
        self.res.f = focal_length
        self.res.init_off_axis = init_off_axis
        self.res.diffuser_angle = diffuser_angle
        self.res.wavelengths = self._get_wl_range()
        self.res.ns = self._sellmeier_polymer(self.res.wavelengths)
        self.res.with_dispersion = with_dispersion
        self.res.n0 = self._sellmeier_polymer(self.res.wl0)
        self.res.rms_height = rms_height
        self.res.diffuser_mask = macro_pixels_phase(self.x, self.y, self.diffuser_angle, rms_height=self.rms_height)
 
    def __getattr__(self, name):
        # This method is only called if 'name' is not found in the instance
        if hasattr(self.res, name):
            return getattr(self.res, name)
        else:
            raise AttributeError(name)


    def _get_wl_range(self):
        """ Return array of N_wl wavelengths spanning +/- Dwl/2 around wl0 equally spaced in frequency. """
        c = 299792458  # speed of light in m/s
        f0 = c / self.wl0
        frange = (c / self.wl0**2) * self.Dwl
        df = frange / self.N_wl
        f = f0 + np.arange(-self.N_wl / 2, self.N_wl / 2) * df
        l = c / f
        return l[::-1]

    def _sellmeier_polymer(self, wls):
        # dispersion curve as provided by RPC photonics (now part of Viavi solutions)
        wls_nm = wls*1e9
        return 1.5375 + 8290.45/wls_nm**2 - 2.11046e8/wls_nm**4

    def _get_wl_factor(self, wl):
        # We assume the diffusre phases are as measured for wl0 which has n0
        factor = self.wl0 / wl # longer wavelength gets less phase
        if self.with_dispersion:
            index = np.where(self.wavelengths == wl)[0][0]
            factor *= self.ns[index] / self.n0  # higher index gets more phase
        return factor

    def make_detection_gaussian(self, lam):
        """ Make a Gaussian beam at the detection plane with self.waist. """
        r2 = (self.XX-self.init_off_axis)**2 + self.YY**2
        E = np.exp(-r2 / (self.waist**2), dtype=np.complex128)
        E /= np.sqrt((np.abs(E)**2).sum())
        return Field(self.x, self.y, lam, E)

    def run_classical_simulation(self, populate_res=True):
        i_ref = 0
        # get classical initial field at crystal plane, to be fair with spot size compared to the SPDC exp.
        # Using the minimal (maximal?) wavelength, since this will result with the same global grid for classical and SPDC exp.
        field_det = self.make_detection_gaussian(self.wavelengths.max())
        field_init = prop_farfield_fft(field_det, self.f)

        fields = []
        delta_lambdas = []
        xs = []
        ys = []

        for wl in self.wavelengths:
            field_crystal = Field(field_init.x.copy(), field_init.y.copy(), wl, field_init.E.copy())
            field_crystal.E *= np.exp(1j * self.diffuser_mask * self._get_wl_factor(wl))
            field_det_new = prop_farfield_fft(field_crystal, self.f)

            delta_lambdas.append(wl - self.wavelengths[i_ref])
            fields.append(field_det_new)
            xs.append(field_det_new.x)
            ys.append(field_det_new.y)

        self.res._classical_fields_E = np.array([f.E.astype(np.complex64) for f in fields])
        self.res._classical_fields_wl = np.array([f.wl for f in fields])
        self.res.classical_delta_lambdas = np.array(delta_lambdas)
        self.res.classical_xs = np.array(xs)
        self.res.classical_ys = np.array(ys)
        if populate_res:
            print("Populating classical")
            self.res._populate_res_classical()

        return self.res


    def run_SPDC_simulation(self, populate_res=True):
        """ returns list of output fields and one-sided delta lambdas"""
        self.res.SPDC_ff_method = 'fft'
        fields = []
        delta_lambdas = []
        xs = []
        ys = []

        for i, wl1 in enumerate(self.wavelengths):
            wl2 = self.wavelengths[len(self.wavelengths)-i-1]
            field_det = self.make_detection_gaussian(wl1)
            field_crystal = prop_farfield_fft(field_det, self.f)
            # TODO: verify sellemier logic
            field_crystal.E *= np.exp(1j * self.diffuser_mask * self._get_wl_factor(wl1))

            # We assume perfect phase matching 
            field_crystal.wl = wl2
            field_crystal.E *= np.exp(1j * self.diffuser_mask * self._get_wl_factor(wl2))
            field_det_new = prop_farfield_fft(field_crystal, self.f)

            delta_lambdas.append(np.abs(wl2-wl1))
            fields.append(field_det_new)
            xs.append(field_det_new.x)
            ys.append(field_det_new.y)

        self.res._SPDC_fields_E = np.array([f.E.astype(np.complex64) for f in fields])
        self.res._SPDC_fields_wl = np.array([f.wl for f in fields])
        self.res.SPDC_delta_lambdas = np.array(delta_lambdas)
        self.res.SPDC_xs = np.array(xs)
        self.res.SPDC_ys = np.array(ys)
        if populate_res:
            print("Populating SPDC")
            self.res._populate_res_SPDC()
        return self.res
