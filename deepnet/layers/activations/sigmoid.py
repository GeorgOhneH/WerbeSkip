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
        return expit(z)

    def forward_backpropagation(self, z):
        self.z = z
        return self.forward(z)

    def make_delta(self, delta):
        z = self.forward(self.z) * (1 - self.forward(self.z))
        delta = delta * z
        return delta
