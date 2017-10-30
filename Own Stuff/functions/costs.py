import numpy as np


class QuadraticCost(object):
    @staticmethod
    def function(a, y):
        return np.sum(0.5 * np.linalg.norm(a - y, axis=0) ** 2) / y.shape[1]

    @staticmethod
    def delta(a, y):
        return a - y
