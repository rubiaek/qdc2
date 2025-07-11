import numpy as np
import datetime
from colorsys import hls_to_rgb
import IPython


# https://github.com/wavefrontshaping/pyMMF/blob/master/pyMMF/functions.py
def colorize(z, theme='dark', saturation=1., beta=1.4, transparent=False, alpha=1., max_threshold=1.):
    r = np.abs(z)
    r /= max_threshold * np.max(np.abs(r))
    arg = np.angle(z)

    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 1. / (1. + r ** beta) if theme == 'white' else 1. - 1. / (1. + r ** beta)
    s = saturation

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    if transparent:
        a = 1. - np.sum(c ** 2, axis=-1) / 3
        alpha_channel = a[..., None] ** alpha
        return np.concatenate([c, alpha_channel], axis=-1)
    else:
        return c


def tnow():
    return datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


def show(fig):
    try:
        shell = IPython.get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return
    except:
        pass
    fig.show()