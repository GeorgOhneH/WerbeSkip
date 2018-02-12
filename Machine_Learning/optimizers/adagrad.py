from .optimizer import Optimizer

import numpy as np


class AdaGrad(Optimizer):
    def __init__(self, learning_rate, e=1e-8):
        super().__init__(learning_rate)
        self.e = e
        self.cache_w = 0
        self.cache_b = 0

    def calculate_change(self, nabla_w, nabla_b):
        self.cache_w += np.power(nabla_w, 2)
        self.cache_b += np.power(nabla_b, 2)
        change_w = np.multiply(self.learning_rate / (np.sqrt(self.cache_w) + self.e), nabla_w)
        change_b = np.multiply(self.learning_rate / (np.sqrt(self.cache_b) + self.e), nabla_b)
        return change_w, change_b
