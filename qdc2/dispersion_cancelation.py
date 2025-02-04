import matplotlib.pyplot as plt
from qdc2.many_wl_fiber import ManyWavelengthFiber


def main():
    fiber_L = 5e6
    mode_mixing = 0
    dz = 50

    s = ManyWavelengthFiber(wl0=0.810, Dwl=0.080, N_wl=161, fiber_L=0.4e6)
    # s.run_PCCs_different_dz(dzs=(0, 20, 40, 60, 150))
    s.run_PCCs_different_dz(dzs=(0, 20, 80), N_classical=5, N_klyshko=3)


if __name__ == "__main__":
    main()

plt.show()

#
# f = s.fibers[0]
# n = 2
# print(f'Getting classical with average on {n}...')
# a, b = s.get_classical_PCCs_average(n)
# print(f'Getting Klyshko with average on {n}......')
# c, d = s.get_klyshko_PCCs_average(n, dz=dz)
# s.show_PCC_classical_and_quantum(a, b, c, d, fiber_L, mode_mixing, dz)
