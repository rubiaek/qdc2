import numpy as np
import matplotlib.pyplot as plt

class Field:
    """
    Simple container for a 2D field:
      - E: 2D complex ndarray
      - x, y: 1D coordinate arrays
      - lam: wavelength in meters
      - k: wave number = 2*pi/lam
    """
    def __init__(self, x, y, wl, E=None):
        self.x = x
        self.y = y
        self.wl = wl
        self.k = 2.0*np.pi / wl
        if E is None:
            Nx = len(x)
            Ny = len(y)
            self.E = np.zeros((Ny, Nx), dtype=np.complex128)
        else:
            self.E = E

    @property
    def dx(self):
        return self.x[1] - self.x[0]

    @property
    def dy(self):
        return self.y[1] - self.y[0]

    def show(self, mode='intensity', xscale=1e6, yscale=1e6, cmap='viridis', title=None):
        """
        Display the field.
          - mode: 'intensity' | 'amplitude' | 'phase'
          - xscale, yscale: coordinate scaling (default 1e3 => mm if x,y in m)
          - cmap: colormap
          - title: optional figure title

        Example usage:
            field.show(mode='intensity', xscale=1e6, yscale=1e6, title='Detection Plane')
        """
        if mode not in ['intensity', 'amplitude', 'phase']:
            raise ValueError("mode must be one of: 'intensity', 'amplitude', 'phase'")

        Xmin, Xmax = self.x[0], self.x[-1]
        Ymin, Ymax = self.y[0], self.y[-1]

        # Prepare data
        if mode == 'intensity':
            data = np.abs(self.E)**2
            label = 'Intensity'
        elif mode == 'amplitude':
            data = np.abs(self.E)
            label = 'Amplitude'
        else:  # 'phase'
            data = np.angle(self.E)
            label = 'Phase [rad]'

        fig, ax = plt.subplots()
        im = ax.imshow(
            data,
            origin='lower',
            extent=[Xmin*xscale, Xmax*xscale, Ymin*yscale, Ymax*yscale],
            cmap=cmap
        )
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(label)

        if title is None:
            title = f"{mode.capitalize()} (λ={self.wl * 1e9:.1f} nm)"
        ax.set_title(title)

        # Axis labels, assuming xscale, yscale: e.g. 1e3 -> "mm"
        xunits = 'mm' if (xscale == 1e3) else 'µm' if (xscale == 1e6) else ''
        yunits = xunits
        ax.set_xlabel(f"x [{xunits}]")
        ax.set_ylabel(f"y [{yunits}]")

        fig.show()
        return fig, ax