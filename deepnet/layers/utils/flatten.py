from layers import Layer

import numpy as np


class Flatten(Layer):
    def __init__(self):
        self.a = None

    def init(self, neurons_before, optimizer):
        x = np.array(neurons_before)
        return np.prod(x)

    def forward(self, a):
        mini_batch_size = a.shape[0]
        out = a.ravel().reshape((mini_batch_size, -1))
        return out

    def forward_backpropagation(self, a):
        self.a = a
        return self.forward(a)

    def make_delta(self, delta):
        out = delta.reshape(self.a.shape)
        return out
