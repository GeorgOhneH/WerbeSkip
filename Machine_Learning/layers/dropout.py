from layers import Layer

import numpy as np


class Dropout(Layer):
    def __init__(self, dropout):
        self.dropout = dropout
        self.dropout_mask = None
        self.z = None
        self.a = None

    def forward(self, a):
        a *= self.dropout
        return a

    def forward_backpropagation(self, a):
        self.dropout_mask = np.random.binomial(1, self.dropout, size=a.shape)
        a = np.multiply(a, self.dropout_mask)
        return a

    def make_delta(self, delta):
        delta = np.multiply(delta, self.dropout_mask)
        return delta

