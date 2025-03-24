import numpy as np
import copy
from qdc.diffuser.field import Field
from functools import cache, lru_cache
import pyfftw
pyfftw.interfaces.cache.enable()
pyfftw.config.NUM_THREADS = 4  # or the number of cores you wish to use


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

def prop_farfield_fft(field, focal_length, lens_phases=False):
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
    if lens_phases:
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


@cache
def get_prop_mat(shape_0, shape_1, dx, dy, wl, dz):
    freq_x = np.fft.fftfreq(shape_1, d=dx)
    freq_y = np.fft.fftfreq(shape_0, d=dy)
    freq_Xs, freq_Ys = np.meshgrid(freq_x, freq_y)

    light_k = 2 * np.pi / wl
    k_x = freq_Xs * 2 * np.pi
    k_y = freq_Ys * 2 * np.pi

    k_z_sqr = light_k**2 - (k_x**2 + k_y**2)
    # clamp negative => evanescent
    np.maximum(k_z_sqr, 0, out=k_z_sqr)
    k_z = np.sqrt(k_z_sqr)
    return np.exp(1j * k_z * dz)


@lru_cache(maxsize=None)
def get_fftw_plans(shape, dtype_str, nthreads=4):
    """
    Create and cache FFTW plan objects for a given array shape and dtype.
    We pass the dtype as a string (e.g. 'complex128') so that it can be cached.
    Returns:
      fft_object: FFTW plan for forward FFT (a -> b)
      ifft_object: FFTW plan for inverse FFT (b -> a)
      a: Aligned input array for FFTW (used for forward FFT and output of inverse FFT)
      b: Aligned output array for FFTW (used for forward FFT and input of inverse FFT)
    """
    # Convert string back to dtype
    dtype = np.dtype(dtype_str)
    a = pyfftw.empty_aligned(shape, dtype=dtype)
    b = pyfftw.empty_aligned(shape, dtype=dtype)
    fft_object = pyfftw.FFTW(a, b,
                             axes=(0, 1),
                             direction='FFTW_FORWARD',
                             flags=('FFTW_MEASURE',),
                             threads=nthreads)
    ifft_object = pyfftw.FFTW(b, a,
                              axes=(0, 1),
                              direction='FFTW_BACKWARD',
                              flags=('FFTW_MEASURE',),
                              threads=nthreads)
    return fft_object, ifft_object, a, b


def propagate_free_space(f: Field, dz, fast=True) -> Field:
    if fast:
        shape = f.E.shape
        dtype_str = str(f.E.dtype)
        fft_object, ifft_object, a, b = get_fftw_plans(shape, dtype_str,
                                                       nthreads=pyfftw.config.NUM_THREADS)
        # Copy input data and execute FFT
        a[:] = f.E  # don't use np.copyto!
        fft_object.execute()  # Performs a -> b

        # Apply propagation in frequency domain
        b *= get_prop_mat(shape[0], shape[1], f.dx, f.dy, f.wl, dz)

        # Perform inverse FFT
        ifft_object.execute()  # Performs b -> a

        # Create output field
        f2 = copy.deepcopy(f)
        f2.E = (a / (shape[0] * shape[1])).copy()
        return f2
    else:
        fa = np.fft.fft2(f.E)
        fa *= get_prop_mat(f.E.shape[0], f.E.shape[1], f.dx, f.dy, f.wl, dz)
        out_E = np.fft.ifft2(fa)

    # Return a new Field with the propagated field.
    f2 = copy.deepcopy(f)
    f2.E = out_E
    return f2
