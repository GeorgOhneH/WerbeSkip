from layers.layer import Layer

import numpy as np


class ReLU(Layer):
    """
    ReLU is an activation function
    It behaves like a layer
    The shape of the input is the same as the output
    """
    def __init__(self):
        self.z = None

    def forward(self, z):
        a = np.maximum(z, 0.0)
        return a

    def forward_backpropagation(self, z):
        self.z = z
        a = np.maximum(z, 0.0)
        return a

    def make_delta(self, delta):
        self.z[self.z <= 0] = 0
        self.z[self.z > 0] = 1
        delta = delta * self.z
        return delta
