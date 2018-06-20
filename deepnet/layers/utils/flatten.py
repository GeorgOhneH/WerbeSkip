from layers import Layer
from utils import flatten, unflatten

import numpywrapper as np


class Flatten(Layer):
    def __init__(self):
        self.a = None

    def init(self, neurons_before, optimizer):
        x = np.array(neurons_before)
        return np.prod(x)

    def forward(self, a):
        return flatten(a)

    def forward_backpropagation(self, a):
        self.a = a
        return self.forward(a)

    def make_delta(self, delta):
        return unflatten(delta, self.a.shape)
