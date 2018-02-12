from .optimizer import Optimizer

import numpy as np


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, e=1e-8):
        super().__init__(learning_rate)
        self.beta2 = beta2
        self.beta1 = beta1
        self.e = e
        self.t = 1
        self.momentum_w = 0
        self.momentum_b = 0
        self.velocity_w = 0
        self.velocity_b = 0

    def calculate_change(self, nabla_w, nabla_b):
        self.momentum_w = self.beta1*self.momentum_w + (1-self.beta1)*nabla_w
        self.momentum_b = self.beta1*self.momentum_b + (1-self.beta1)*nabla_b

        self.velocity_w = self.beta2*self.velocity_w + (1-self.beta2)*np.power(nabla_w, 2)
        self.velocity_b = self.beta2*self.velocity_b + (1-self.beta2)*np.power(nabla_b, 2)

        estimate_momentum_w = self.momentum_w / (1 - self.beta1 ** self.t)
        estimate_momentum_b = self.momentum_b / (1 - self.beta1 ** self.t)

        estimate_velocity_w = self.velocity_w / (1 - self.beta2 ** self.t)
        estimate_velocity_b = self.velocity_b / (1 - self.beta2 ** self.t)

        change_w = self.learning_rate * estimate_momentum_w / (np.sqrt(estimate_velocity_w) + self.e)
        change_b = self.learning_rate * estimate_momentum_b / (np.sqrt(estimate_velocity_b) + self.e)

        self.t += 1

        return change_w, change_b
