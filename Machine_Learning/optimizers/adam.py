from optimizers.optimizer import Optimizer

import numpy as np


class Adam(Optimizer):
    """
    Adam is like RMSprop but with momentum

    It works really well
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, e=1e-8):
        super().__init__(learning_rate)
        self.beta2 = beta2
        self.beta1 = beta1
        self.e = e
        self.t = 1
        self.momenta = None
        self.velocities = None

    def calculate_change(self, *nablas):
        change = []

        if self.momenta is None and self.velocities is None:
            self.momenta = [0 for _ in nablas]
            self.velocities = [0 for _ in nablas]

        for index, nabla in enumerate(nablas):
            self.momenta[index] = self.beta1*self.momenta[index] + (1-self.beta1)*nabla
            self.velocities[index] = self.beta2*self.velocities[index] + (1-self.beta2)*np.power(nabla, 2)

            estimate_momentum = self.momenta[index] / (1 - self.beta1 ** self.t)
            estimate_velocity = self.velocities[index] / (1 - self.beta2 ** self.t)

            change.append(self.learning_rate * estimate_momentum / (np.sqrt(estimate_velocity) + self.e))

        self.t += 1

        return change
