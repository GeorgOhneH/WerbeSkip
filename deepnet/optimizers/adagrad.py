from optimizers.optimizer import Optimizer

import numpywrapper as np


class AdaGrad(Optimizer):
    """
    AdaGrad computes the gradient by taking the
    the previous gradients into account
    """
    def __init__(self, learning_rate, e=1e-8):
        super().__init__(learning_rate)
        self.e = e
        self.caches = None

    def calculate_change(self, *nablas):
        changes = []

        if self.caches is None:
            self.caches = [0 for _ in nablas]

        for index, nabla in enumerate(nablas):
            self.caches[index] += nabla ** 2
            changes.append(self.learning_rate / (np.sqrt(self.caches[index]) + self.e) * nabla)

        return changes
