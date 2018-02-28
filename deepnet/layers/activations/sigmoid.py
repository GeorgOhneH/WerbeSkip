from layers.layer import Layer

import numpy as np
from scipy.special import expit


class Sigmoid(Layer):
    """
    Sigmoid is an activation function
    It behaves like a layer
    The shape of the input is the same as the output
    """
    def __init__(self):
        self.z = None

    def forward(self, z):
        a = expit(z)
        return a

    def forward_backpropagation(self, z):
        self.z = z
        a = expit(z)
        return a

    def make_delta(self, delta):
        z = np.multiply(self.forward(self.z), (1 - self.forward(self.z)))
        delta = delta * z
        return delta
