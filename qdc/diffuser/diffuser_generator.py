import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def wrapped_phase_diffuser(x, y, ref_wl, OPD_range, corr_length):
    """
    Generate a phase diffuser by first creating a smooth height map
    with a specified OPD_range (in length units) and correlation length,
    then converting it to a phase and wrapping it modulo 2pi.

    Parameters
    ----------
    x, y : 1D arrays
        Spatial coordinates (assumed uniformly spaced).
    ref_wl : float
        Reference wavelength.
    OPD_range : float
        Desired range (peak-to-peak) of the underlying optical path difference. Units of wl.
    corr_length : float
        Correlation length (controls local slopes) in the same units as x and y.

    Returns
    -------
    phase_wrapped : 2D array
        Phase mask wrapped to [0, 2pi) at the reference wavelength.
    """
    Nx = len(x)
    Ny = len(y)
    dx = x[1] - x[0]

    # Generate a random height field (white noise)
    h = np.random.randn(Ny, Nx)

    # Smooth it so that local gradients are small; sigma in pixels:
    sigma_pixels = corr_length / dx
    h_smooth = gaussian_filter(h, sigma=sigma_pixels)

    # Normalize to the desired range (peak-to-peak roughly equal to OPD_range)
    h_min, h_max = h_smooth.min(), h_smooth.max()
    h_scaled = (h_smooth - h_min) / (h_max - h_min) * OPD_range * ref_wl

    # Convert to phase (OPD->phase at ref wavelength)
    phase = (2 * np.pi / ref_wl) * h_scaled

    # Wrap phase to [0, 2pi)
    phase_wrapped = np.mod(phase, 2 * np.pi)
    return phase_wrapped


def phase_screen_diff_rfft(x, y, ref_wl, theta, rms_height=5):
    """
    Generate a real-valued phase screen for a thin diffuser mask.

    The phase screen is defined at a reference wavelength 'ref_wl' and has
    an RMS phase variation (in units of wavelength) given by 'rms_height'.
    The scattering angle 'theta' (in radians) sets the width of the
    power spectral density (PSD). (You may need to adjust units and the PSD
    profile to suit your application.)

    Parameters
    ----------
    x : 1D numpy array
        x coordinates (assumed uniformly spaced).
    y : 1D numpy array
        y coordinates (assumed uniformly spaced).
    ref_wl : float
        Reference wavelength (same length units as x and y).
    theta : float
        Characteristic scattering angle (in radians).
    rms_height : float, optional
        RMS phase variation in units of wavelength (default is 5).

    Returns
    -------
    phase_ref : 2D numpy array
        Phase screen evaluated on the (x, y) grid at the reference wavelength.
        For a different wavelength lam, use:
            phase_lam = (ref_wl / lam) * phase_ref
        since the phase is proportional to (OPD / lam).
    """
    Nx = len(x)
    Ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # reference wavenumber (in units 1/length)
    k_ref = 2 * np.pi / ref_wl

    # The cutoff (standard deviation) in Fourier space is set by theta.
    # (Depending on your exact definition, you might adjust the factor below.)
    sigma = (theta * k_ref) / (2 * np.pi)

    # Get the Fourier space coordinates for a real FFT.
    # For the x-direction (last axis) use rfft frequencies.
    fx = np.fft.rfftfreq(Nx, d=dx)  # shape: (Nx//2 + 1,)
    # For the y-direction use full fft frequencies.
    fy = np.fft.fftfreq(Ny, d=dy)  # shape: (Ny,)
    # Create a meshgrid. With 'xy' indexing the output arrays have shape (Ny, Nx//2+1).
    Fx, Fy = np.meshgrid(fx, fy, indexing='xy')

    # Define a radially symmetric PSD (here a Gaussian; you can choose another shape)
    PSD = np.exp(- (Fx ** 2 + Fy ** 2) / (2 * sigma ** 2))
    PSD[0, 0] = 0  # Optionally remove the DC component

    # Generate random Fourier components:
    # - Assign a random phase uniformly distributed from 0 to 2pi.
    # - The amplitude is set to sqrt(PSD).
    random_phase = 2 * np.pi * np.random.rand(*PSD.shape)
    amplitude = np.sqrt(PSD)
    spectrum = amplitude * np.exp(1j * random_phase)

    # Inverse real FFT to get the phase screen in real space.
    phase_ref = np.fft.irfft2(spectrum, s=(Ny, Nx))

    # Normalize the resulting phase screen so that its RMS equals rms_height.
    current_rms = np.sqrt(np.mean(phase_ref ** 2))
    if current_rms < 1e-12:
        raise ValueError("Computed RMS is too small; check your parameters.")
    phase_ref *= (rms_height / current_rms)

    return phase_ref


def phase_screen_diff(x, y, ref_wl, theta, rms_height=5):
    """
    ** rms_height in units of wavelength **

    Generate a random 2D phase screen for a reference wavelength lam_ref,
    using a von Kármán-like PSD with scattering angle = theta.

    Returns a real 2D array 'phase_ref(x,y)' that is the phase at lam_ref.
    If you want the phase for lam != lam_ref, use:

        phase_lam = (lam_ref / lam) * phase_ref

    Because OPD is fixed by surface height, so phase ~ (2*pi / lam)*OPD.
    """
    Nx = len(x)
    Ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    assert Nx == Ny, "For simplicity, assume Nx=Ny in this example."

    k_ref = 2 * np.pi / ref_wl
    sigma = (theta * k_ref) / (2*np.pi)

    # Frequency coords
    df_x = 1/(Nx*dx)
    df_y = 1/(Ny*dy)
    fx = np.arange(-Nx/2, Nx/2)*df_x
    fy = np.arange(-Ny/2, Ny/2)*df_y
    Fx, Fy = np.meshgrid(fx, fy, indexing='xy')

    # PSD
    PSD = np.exp(-(Fx**2 + Fy**2)/(2*sigma**2))
    PSD[Nx//2, Ny//2] = 0  # remove DC

    # Random complex
    rand_complex = (np.random.randn(Ny, Nx) + 1j*np.random.randn(Ny, Nx))
    spectrum = rand_complex * np.sqrt(PSD)

    # Inverse FFT -> real-space random complex
    screen_complex = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum)))
    phase_ref = np.real(screen_complex)  # Ohads original had here np.angle and did not have the rms scaling
    current_rms = np.sqrt(np.mean(phase_ref**2))
    assert current_rms > 1e-20
    phase_ref *= (rms_height / current_rms)

    return phase_ref


def grating_phase(x, y, ref_wl, theta):
    XX, YY = np.meshgrid(x, y)
    d = ref_wl / np.sin(theta)  # grating period, d*sin(theta)=lambda*m; m=1
    phi = (2 * np.pi / d * XX) % (2 * np.pi)
    return phi


def macro_pixels_phase(x, y, theta, rms_height):
    Dx = x[-1] - x[0]
    Nx = len(x)
    macro_pixel_size = 1e-6 / theta
    macro_pixels_N = int(Dx / macro_pixel_size)
    A = np.random.rand(macro_pixels_N, macro_pixels_N)
    A2 = cv2.resize(A, (Nx, Nx), interpolation=cv2.INTER_NEAREST)
    current_rms = np.sqrt(np.mean(A2 ** 2))
    A3 = A2 * (rms_height / current_rms)
    return A3
