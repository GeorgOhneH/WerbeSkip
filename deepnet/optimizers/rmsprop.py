from deepnet.optimizers import Optimizer

import numpywrapper as np


class RMSprop(Optimizer):
    """
    RMSprop is similar to AdaGrad
    The difference is that it has a decay rate, so that
    the previous gradient are not able to slow down the learning rate
    as strong as they are able by AdaGrad
    """
    def __init__(self, learning_rate=0.001, decay_rate=0.9, e=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.e = e
        self.caches = None

    def __str__(self):
        return "{} decay_rate: {}".format(super(RMSprop, self).__str__(), self.decay_rate)

    def calculate_change(self, *nablas):
        changes = []

        if self.caches is None:
            self.caches = [0 for _ in nablas]

        for index, nabla in enumerate(nablas):
            self.caches[index] = nabla ** 2 * self.decay_rate + (1-self.decay_rate) * self.caches[index]
            changes.append(self.learning_rate / (np.sqrt(self.caches[index]) + self.e) * nabla)

        return changes
