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
    2D Inverse FFT -> g(x, y), given G(k_x, k_y) and freq coords f_x,f_y.
    """
    Nx = len(f_x)
    Ny = len(f_y)
    df_x = f_x[1] - f_x[0]
    df_y = f_y[1] - f_y[0]

    g = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G))) * (Nx * df_x) * (Ny * df_y)
    return g

def prop_farfield_fft(field, focal_length):
    """
    Rigorous thin-lens Fraunhofer propagation from a plane at distance f in front of the lens
    to the back focal plane at distance f.

    field: Field object with .E, .x, .y, .lam, .k
    1) Multiply by lens phase factor: exp[-i * k / (2*f) * (x^2 + y^2)]
    2) Compute 2D FFT
    3) Apply standard Fraunhofer scaling:
       - new_x = lam * f * f_x
       - new_y = lam * f * f_y
    4) Optional amplitude factor:  ( e^{i k f} / (i lam f} ), ignoring global phase
       but we keep 1/(i lam f) as amplitude scaling if you want the physically correct magnitude.
    """
    f2 = copy.deepcopy(field)

    # 1) Lens phase
    XX, YY = np.meshgrid(f2.x, f2.y, indexing='xy')
    lens_phase = np.exp(-1j * (f2.k / (2*focal_length)) * (XX**2 + YY**2))
    f2.E *= lens_phase

    # 2) FFT
    G, f_x, f_y = ft2(f2.E, f2.x, f2.y)

    # 3) Rescale coordinates: x' = lam * f * f_x
    lam = f2.lam
    new_x = lam * focal_length * f_x
    new_y = lam * focal_length * f_y

    # 4) Optional amplitude factor: 1/(i * lam * f)
    #    You can include a global phase e^{i k f} if desired.
    #    For now, we just do amplitude scaling to get physically correct intensities.
    G *= (1.0 / (1j * lam * focal_length))

    f2.E = G
    f2.x = new_x
    f2.y = new_y
    return f2

def backprop_farfield_fft(field, focal_length):
    """
    Same as prop_farfield_fft but used in the Klyshko picture to denote 'backwards' propagation.
    Mathematically it's similar, except you might define a conjugate or negative focal length.
    We'll keep it identical here for simplicity.
    """
    return prop_farfield_fft(field, focal_length)


def phase_screen_diff(x, y, lam_ref, theta):
    """
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

    k_ref = 2 * np.pi / lam_ref
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
    phase_ref = np.angle(screen_complex)

    return phase_ref
