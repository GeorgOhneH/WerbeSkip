import numpy as np


class Sigmoid(object):
    @staticmethod
    def function(z):
        z = 1.0 / (1.0 + np.exp(-z))
        return z

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
