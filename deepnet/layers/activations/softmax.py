from layers.layer import Layer

import numpy as np


class SoftMax(Layer):
    """
    SoftMax is an activation function
    It behaves like a layer
    The shape of the input is the same as the output

    The method make_delta doesn't do anything, because
    the derivative/delta was already calculated in the
    CrossEntropyCost
    """

    def forward(self, z):
        exps = np.exp(z - np.max(z, axis=0))
        return exps / np.sum(exps, axis=0)

    def forward_backpropagation(self, z):
        exps = np.exp(z - np.max(z, axis=0))
        return exps / np.sum(exps, axis=0)

    def make_delta(self, delta):
        return delta
