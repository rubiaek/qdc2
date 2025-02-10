import numpy as np
from qdc.diffuser.diffuser_sim import DiffuserSimulation
from qdc.diffuser.diffuser_result import DiffuserResult


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
    result_map = sim.run_simulation()
    assert result_map.shape == (64, 64), "Output intensity map must match grid size."
    assert np.all(result_map >= 0), "Intensity must be non-negative."

def test_result_contrast():
    test_map = np.ones((10, 10))  # uniform intensity
    res = DiffuserResult(intensity_map=test_map)
    contrast = res.compute_contrast()
    assert np.isclose(contrast, 0), "Contrast of a uniform map should be 0."

# You can add more tests (phase correlation, single-wavelength check, etc.).
