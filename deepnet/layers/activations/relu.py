from deepnet.layers import Layer

import numpywrapper as np


class ReLU(Layer):
    """
    ReLU is an activation function
    It behaves like a layer
    The shape of the input is the same as the output
    """
    def __init__(self):
        self.z = None

    def forward(self, z):
        return np.maximum(z, 0.0)

    def forward_backpropagation(self, z):
        self.z = z
        return self.forward(z)

    def make_delta(self, delta):
        self.z[self.z <= 0] = 0
        self.z[self.z > 0] = 1
        delta = delta * self.z
        return delta
