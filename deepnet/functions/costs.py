import numpy as np


class QuadraticCost(object):
    @staticmethod
    def function(a, y):
        """Computes the cost function"""
        return np.sum(0.5 * np.linalg.norm(a - y, axis=0) ** 2) / y.shape[1]

    @staticmethod
    def delta(a, y):
        """Calculates the derivative"""
        return (a - y) / y.shape[1]


class CrossEntropyCost(object):
    """
    The cross entropy only works with the softmax and can't be used
    without it.
    The delta method is the derivative of the cross entropy AND the
    softmax combined
    """
    @staticmethod
    def function(a, y):
        """Computes the cost function"""
        return -np.sum(y * np.log(a)) / y.shape[1]

    @staticmethod
    def delta(a, y):
        """Calculates the derivative and the derivative of the softmax"""
        return (a - y) / y.shape[1]
