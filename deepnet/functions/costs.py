import numpy as np


class Cost(object):
    """
    Base class of the cost functions
    """
    def __str__(self):
        return self.__class__.__name__


class QuadraticCost(Cost):
    @staticmethod
    def function(a, y):
        """Computes the cost function"""
        return np.sum(0.5 * np.linalg.norm(a - y, axis=1) ** 2) / y.shape[0]

    @staticmethod
    def delta(a, y):
        """Calculates the derivative"""
        return (a - y) / y.shape[0]


class CrossEntropyCost(Cost):
    """
    The cross entropy only works with the softmax and can't be used
    without it.
    The delta method is the derivative of the cross entropy AND the
    softmax combined
    """

    @staticmethod
    def function(a, y):
        """Computes the cost function"""
        return -np.sum(y * np.log(a)) / y.shape[0]

    @staticmethod
    def delta(a, y):
        """Calculates the derivative and the derivative of the softmax"""
        return (a - y) / y.shape[0]
