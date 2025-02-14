import numpy as np
import copy
from qdc.diffuser.field import Field


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
    lam = f2.wl
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


def propagate_free_space(f : Field, dz) -> Field:
    """
    FFT-based free-space propagation by distance dz.
    * E: field (2D or flattened), shape (n, n).
    * dz: propagation distance
    * wavelength: in microns
    * dx: pixel size in x & y (assumed the same in both directions)
    Returns a flattened field (same shape as input).
    """
    fa = np.fft.fft2(f.E)
    freq_x = np.fft.fftfreq(f.E.shape[1], d=f.dx)
    freq_y = np.fft.fftfreq(f.E.shape[0], d=f.dy)
    freq_Xs, freq_Ys = np.meshgrid(freq_x, freq_y)

    light_k = 2 * np.pi / f.wl
    k_x = freq_Xs * 2 * np.pi
    k_y = freq_Ys * 2 * np.pi

    k_z_sqr = light_k**2 - (k_x**2 + k_y**2)
    # clamp negative => evanescent
    np.maximum(k_z_sqr, 0, out=k_z_sqr)
    k_z = np.sqrt(k_z_sqr)

    # free-space phase shift
    fa *= np.exp(1j * k_z * dz)

    out_E = np.fft.ifft2(fa)
    f2 = copy.deepcopy(f)
    f2.E = out_E
    return f2


