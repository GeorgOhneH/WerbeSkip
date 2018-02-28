from layers.layer import Layer

import numpy as np


class LReLU(Layer):
    """
    Leaky ReLU is an activation function
    It behaves like a layer
    The shape of the input is the same as the output
    """
    def __init__(self, alpha=0.01):
        self.z = None
        self.alpha = alpha

    def forward(self, z):
        z[z < 0] *= self.alpha
        return z

    def forward_backpropagation(self, z):
        self.z = z
        return self.forward(z)

    def make_delta(self, delta):
        dz = np.ones_like(self.z)
        dz[self.z < 0] = self.alpha
        delta = delta * dz
        return delta
