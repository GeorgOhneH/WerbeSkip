from layers.layer import Layer

import numpy as np


class FullyConnectedLayer(Layer):
    def init(self, neurons_before):
        self.biases = np.random.randn(self.neurons, 1)
        self.weights = np.random.randn(self.neurons, neurons_before) / np.sqrt(neurons_before)
        return self.neurons

    # Input Matrix Output Matrix
    def forward(self, a):
        z = self.weights @ a + self.biases
        a = self.activation.function(z)
        return a

    def forward_backpropagation(self, a):
        self.a = a
        z = self.weights @ a + self.biases
        self.z = z
        a = self.activation.function(z)
        return a

    def make_delta(self, delta):
        delta = np.multiply(delta, self.activation.derivative(self.z))
        self.nabla_b = np.sum(delta, axis=1)
        self.nabla_w = delta @ self.a.T
        return self.weights.T @ delta

    def adjust_weights(self, factor):
        self.weights -= factor * self.nabla_w
        self.biases -= factor * self.nabla_b
