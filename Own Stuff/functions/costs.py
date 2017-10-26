import numpy as np


class QuadraticCost(object):
    @staticmethod
    def function(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(a, y):
        return a - y