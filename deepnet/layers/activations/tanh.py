from layers.layer import Layer

import numpy as np


class TanH(Layer):
    """
    TanH is an activation function
    It behaves like a layer
    The shape of the input is the same as the output
    """
    def __init__(self):
        self.z = None

    def forward(self, z):
        a = np.tanh(z)
        return a

    def forward_backpropagation(self, z):
        self.z = z
        a = np.tanh(z)
        return a

    def make_delta(self, delta):
        z = 1 - self.forward(self.z) ** 2
        delta = delta * z
        return delta
