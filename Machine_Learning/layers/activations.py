from .layer import Layer

import numpy as np
from scipy.special import expit


class ReLU(Layer):
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
        delta = np.multiply(delta, self.z)
        return delta


class Sigmoid(Layer):
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
        delta = np.multiply(delta, z)
        return delta
