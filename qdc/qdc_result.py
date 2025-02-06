import copy
import numpy as np
import matplotlib.pyplot as plt


class QDCResult(object):
    def __init__(self):
        self.classical_PCCs = None
        self.two_ph_PCCs = None


    def saveto(self, path):
        d = copy.deepcopy(self.__dict__)
        np.savez(path, **d)

    def loadfrom(self, path):
        data = np.load(path, allow_pickle=True)
        self.__dict__.update(data)
