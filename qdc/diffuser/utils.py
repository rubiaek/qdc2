import numpy as np
import copy

def ft2(g, x, y):
    """
    2D Forward FFT -> G(k_x, k_y), returning the frequency coordinates too.
    Assumes x[len(x)//2] ~ 0 for symmetrical grid about 0.
    """
    Nx = len(x)
    Ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    G = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g))) * dx * dy

    # Frequency coordinates
    df_x = 1.0 / (Nx * dx)
    df_y = 1.0 / (Ny * dy)
    f_x = np.arange(-Nx/2, Nx/2) * df_x
    f_y = np.arange(-Ny/2, Ny/2) * df_y
    return G, f_x, f_y

def ift2(G, f_x, f_y):
    """
    2D Inverse FFT -> g(x, y) given G in frequency space and freq coords f_x,f_y.
    """
    Nx = len(f_x)
    Ny = len(f_y)
    df_x = f_x[1] - f_x[0]
    df_y = f_y[1] - f_y[0]

    g = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G))) * (Nx*df_x) * (Ny*df_y)
    return g

def prop_farfield_fft(field, focal_length):
    """
    Far-field (Fraunhofer) propagation for a thin lens of focal_length.
    - field is a Field object with .E, .x, .y, .k
    - This performs an FFT, then remaps the output grid to real-space coordinates:
        x' = k_x * (focal_length / k)
        y' = k_y * (focal_length / k)
    Returns a new Field with the propagated field and updated x,y grids.
    """
    f2 = copy.deepcopy(field)

    G, f_x, f_y = ft2(f2.E, f2.x, f2.y)
    k_x = f_x * 2.0 * np.pi
    k_y = f_y * 2.0 * np.pi

    # New real-space coordinates after lens
    new_x = k_x * (focal_length / f2.k)
    new_y = k_y * (focal_length / f2.k)

    f2.E = G
    f2.x = new_x
    f2.y = new_y
    return f2

def backprop_farfield_fft(field, focal_length):
    """
    Same as prop_farfield_fft but conceptually 'backwards' in the Klyshko picture.
    In practice, it’s just an FFT lens transform.
    You could define sign conventions if needed.
    """
    # Numerically it's identical to forward, except you might choose a
    # conjugate or negative focal length, but for a simple advanced-wave
    # approach, we can reuse the same routine.
    return prop_farfield_fft(field, focal_length)


def phase_screen_diff(x, y, lam, theta):
    """
    Generates a random 2D phase screen based on a von Karman-like PSD:
      - x,y are spatial coordinates (1D arrays) of length N
      - lam is wavelength (meters)
      - theta is scattering angle (radians)
    Returns a 2D array of phase (real-space) with zero-mean in angle() sense.

    Adapted from the MATLAB snippet:
      sigma = (theta * k) / (2*pi),  with k = 2*pi / lam.
      PSD = exp(-(Fx^2 + Fy^2)/(2*sigma^2))
    Then the random screen is the angle of the inverse FFT of a random spectrum
    weighted by sqrt(PSD).
    """
    Nx = len(x)
    Ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    assert Nx == Ny, "For simplicity, assume Nx=Ny in this example."

    k = 2*np.pi / lam
    sigma = (theta * k) / (2*np.pi)

    # Frequency coords
    df_x = 1/(Nx*dx)
    df_y = 1/(Ny*dy)
    fx = np.arange(-Nx/2, Nx/2)*df_x
    fy = np.arange(-Ny/2, Ny/2)*df_y
    Fx, Fy = np.meshgrid(fx, fy, indexing='xy')

    # PSD
    PSD = np.exp(-(Fx**2 + Fy**2)/(2*sigma**2))
    # Zero out DC component
    PSD[Nx//2, Ny//2] = 0

    # Random complex
    rand_complex = (np.random.randn(Ny, Nx) + 1j*np.random.randn(Ny, Nx))
    spectrum = rand_complex * np.sqrt(PSD)

    # Inverse FFT -> real-space random
    screen_complex = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum)))
    phase_screen = np.angle(screen_complex)

    return phase_screen
