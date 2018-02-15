from layers.layer import Layer

import numpy as np
from scipy.special import expit


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
