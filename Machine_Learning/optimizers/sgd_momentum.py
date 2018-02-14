from .optimizer import Optimizer

import numpy as np


class SGDMomentum(Optimizer):
    """
    This is similar to stochastic gradient descent
    The different is that is also has a momentum and velocity

    :param nesterov: bool
    The nesterov flag prevents that the error goes to fast in one direction
    by looking ahead
    """
    def __init__(self, learning_rate, momentum=0.9, nesterov=False):
        super().__init__(learning_rate)
        self.nesterov = nesterov
        self.momentum = momentum
        self.velocities = None

    def calculate_change(self, *nablas):
        nablas = list(nablas)
        if self.velocities is None:
            self.velocities = [0 for _ in nablas]

        if self.nesterov:
            for index, velocity in enumerate(self.velocities):
                nablas[index] += self.momentum * velocity

        for index, nabla in enumerate(nablas):
            self.velocities[index] = self.momentum*self.velocities[index] + self.learning_rate*nabla

        return self.velocities
