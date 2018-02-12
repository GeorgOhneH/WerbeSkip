from .optimizer import Optimizer

import numpy as np


class AdaGrad(Optimizer):
    def __init__(self, learning_rate, e=1e-8):
        super().__init__(learning_rate)
        self.e = e
        self.caches = None

    def calculate_change(self, *nablas):
        changes = []

        if self.caches is None:
            self.caches = [0 for _ in nablas]

        for index, nabla in enumerate(nablas):
            self.caches[index] += np.power(nabla, 2)
            changes.append(np.multiply(self.learning_rate / (np.sqrt(self.caches[index]) + self.e), nabla))

        return changes
