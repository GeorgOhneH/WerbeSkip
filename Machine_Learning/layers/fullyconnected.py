from layers.layer import Layer

import numpy as np


class FullyConnectedLayer(Layer):
    def init(self, neurons_before):
        self.biases = np.random.randn(self.neurons, 1)
        self.weights = np.random.randn(self.neurons, neurons_before) / np.sqrt(neurons_before)

    # Input Matrix Output Matrix
    def forward(self, a):
        z = np.dot(self.weights, a) + self.biases
        a = self.activation.function(z)
        return a

    def forward_backpropagation(self, a):
        self.before_a = a
        z = np.dot(self.weights, a) + self.biases
        a = self.activation.function(z)
        self.z = z
        self.a = a
        return a

    def make_first_delta(self,cost, y):
        delta = np.multiply(cost.delta(self.a, y), self.activation.derivative(self.z))
        self.update_nabla(delta)
        return delta, self.weights

    def make_delta(self, delta, last_weights):
        delta = np.multiply(np.dot(last_weights.transpose(), delta), self.activation.derivative(self.z))
        self.update_nabla(delta)
        return delta, self.weights

    def update_nabla(self, delta):
        self.nabla_b = np.sum(delta, axis=1)
        self.nabla_w = np.dot(delta, self.before_a.transpose())

    def adjust_weights(self, factor):
        self.weights -= np.multiply(factor, self.nabla_w)
        self.biases -= np.multiply(factor, self.nabla_b)
