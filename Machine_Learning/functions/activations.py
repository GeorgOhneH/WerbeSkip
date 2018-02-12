import numpy as np
from scipy.special import expit


class Sigmoid(object):
    @staticmethod
    def function(z):
        return expit(z)

    @staticmethod
    def derivative(z):
        return np.multiply(Sigmoid.function(z), (1 - Sigmoid.function(z)))


class ReLU(object):
    @staticmethod
    def function(z):
        return np.maximum(z, 0.0)

    @staticmethod
    def derivative(z):
        z[z <= 0] = 0
        z[z > 0] = 1
        return z
