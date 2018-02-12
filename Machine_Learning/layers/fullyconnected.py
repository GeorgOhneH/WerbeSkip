from layers.layer import Layer

import numpy as np


class FullyConnectedLayer(Layer):
    def __init__(self, neurons):
        self.neurons = neurons
        self.optimizer = None
        self.biases = None
        self.weights = None
        self.nabla_b = None
        self.nabla_w = None
        self.a = None

    def init(self, neurons_before, optimizer):
        self.biases = np.random.randn(self.neurons, 1)
        self.weights = np.random.randn(self.neurons, neurons_before) / np.sqrt(neurons_before)
        self.optimizer = optimizer
        return self.neurons

    # Input Matrix Output Matrix
    def forward(self, a):
        z = self.weights @ a + self.biases
        return z

    def forward_backpropagation(self, a):
        self.a = a
        z = self.weights @ a + self.biases
        return z

    def make_delta(self, delta):
        self.nabla_b = np.sum(delta, axis=1)
        self.nabla_w = delta @ self.a.T
        return self.weights.T @ delta

    def adjust_weights(self, mini_batch_size):
        change_w, change_b = self.optimizer.calculate_change(self.nabla_w, self.nabla_b)
        self.weights -= change_w/mini_batch_size
        self.biases -= change_b/mini_batch_size
