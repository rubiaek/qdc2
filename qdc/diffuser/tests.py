import numpy as np
from qdc.diffuser.diffuser_sim import DiffuserSimulation
from qdc.diffuser.diffuser_result import DiffuserResult
from qdc.diffuser.utils import phase_screen_diff


def test_gaussian_generation():
    sim = DiffuserSimulation(Nx=64, Ny=64, Lx=1e-3, Ly=1e-3, waist=20e-6)
    E_det = sim.make_detection_gaussian()
    assert E_det.shape == (64, 64), "Gaussian field must match grid size."
    assert np.isclose(E_det.max(), 1.0), "Max amplitude of normalized Gaussian should be ~1."

def test_random_phase():
    sim = DiffuserSimulation(Nx=64, Ny=64)
    phase = sim.make_random_phase_screen(sim.wavelength_c)
    assert phase.shape == (64, 64), "Phase screen must match grid size."
    # Rough check for randomness
    mean_phase = np.mean(phase)
    assert 0 < mean_phase < 2*np.pi, "Mean phase should be between 0 and 2π for uniform random."

def test_run_simulation():
    sim = DiffuserSimulation(Nx=64, Ny=64, Nwl=3)
    result_map = sim.run_SPDC_simulation()
    assert result_map.shape == (64, 64), "Output intensity map must match grid size."
    assert np.all(result_map >= 0), "Intensity must be non-negative."

def test_result_contrast():
    test_map = np.ones((10, 10))  # uniform intensity
    res = DiffuserResult(intensity_map=test_map)
    contrast = res.compute_contrast()
    assert np.isclose(contrast, 0), "Contrast of a uniform map should be 0."

def test_one_phase_screen():
    Nx=64; Ny=64
    x = np.linspace(-1e-3,1e-3,Nx)
    y = np.linspace(-1e-3,1e-3,Ny)
    lam_c = 800e-9
    phase_ref = phase_screen_diff(x, y, lam_c, theta=0.5)
    # For a second lam, ensure phase scale is consistent
    lam2 = lam_c*1.1
    phase2 = (lam_c / lam2)*phase_ref
    assert phase2.shape == phase_ref.shape