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

def show_color_legend_ax(ax, theme='dark'):
    r_vals = np.linspace(0, 1, 256)
    phi_vals = np.linspace(-np.pi, np.pi, 256)
    R, P = np.meshgrid(r_vals, phi_vals)
    Z = R * np.exp(1j * P)
    color_img = colorize(Z, theme=theme)

    im = ax.imshow(
        color_img,
        origin='lower',
        extent=[-np.pi, np.pi, 0, 1],
        aspect='auto'
    )
    ax.set_xlabel("Phase [rad]", fontsize=9)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=8)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Magnitude", fontsize=9)

    # Move ticks and label to the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # Optional: reduce tick size
    ax.tick_params(labelsize=8, length=3)


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