from .optimizer import Optimizer

import numpy as np


class SGDMomentum(Optimizer):
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
# from .optimizer import Optimizer
#
#
# class SGDMomentum(Optimizer):
#     def __init__(self, learning_rate, momentum=0.9, nesterov=False):
#         super().__init__(learning_rate)
#         self.nesterov = nesterov
#         self.momentum = momentum
#         self.velocity_w = 0
#         self.velocity_b = 0
#
#     def calculate_change(self, nabla_w, nabla_b):
#         if self.nesterov:
#             nabla_w += self.momentum * self.velocity_w
#             nabla_b += self.momentum * self.velocity_b
#         self.velocity_w = self.momentum * self.velocity_w + self.learning_rate * nabla_w
#         self.velocity_b = self.momentum * self.velocity_b + self.learning_rate * nabla_b
#         return self.velocity_w, self.velocity_b
