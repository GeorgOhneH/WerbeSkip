from .optimizer import Optimizer

import numpy as np


class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, decay_rate=0.9, e=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.e = e
        self.caches = None

    def calculate_change(self, *nablas):
        changes = []

        if self.caches is None:
            self.caches = [0 for _ in nablas]

        for index, nabla in enumerate(nablas):
            self.caches[index] = np.power(nabla, 2) * self.decay_rate + (1-self.decay_rate) * self.caches[index]
            changes.append(np.multiply(self.learning_rate / (np.sqrt(self.caches[index]) + self.e), nabla))

        return changes
