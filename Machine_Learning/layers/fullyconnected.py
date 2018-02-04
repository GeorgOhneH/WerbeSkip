import numpy as np


class FullyConnectedLayer(object):
    def __init__(self, neurons, activation, dropout):
        self.neurons = neurons
        self.activation = activation
        self.dropout = dropout
        self.dropout_mask = None
        self.biases = None
        self.weights = None
        self.nabla_b = None
        self.nabla_w = None
        self.z = None
        self.before_a = None
        self.a = None

    def init(self, neurons_before):
        self.biases = np.random.randn(self.neurons, 1)
        self.weights = np.random.randn(self.neurons, neurons_before) / np.sqrt(neurons_before)

    # Input Matrix Output Matrix
    def forward(self, a):
        z = np.dot(self.weights, a) + self.biases
        a = self.activation.function(z)
        if self.dropout is not None:
            a *= self.dropout
        return a

    def forward_backpropagation(self, a):
        self.before_a = a
        z = np.dot(self.weights, a) + self.biases
        a = self.activation.function(z)
        if self.dropout is not None:
            self.dropout_mask = np.random.binomial(1, self.dropout, size=a.shape)
            a = np.multiply(a, self.dropout_mask)
        self.z = z
        self.a = a
        return a

    def make_first_delta(self, cost, y):
        delta = np.multiply(cost.delta(self.a, y), self.activation.derivative(self.z))
        self.update_nabla(delta)
        return delta

    def make_next_delta(self, delta, last_weights):
        delta = np.multiply(np.dot(last_weights.transpose(), delta), self.activation.derivative(self.z))
        if self.dropout is not None:
            delta = np.multiply(delta, self.dropout_mask)
        self.update_nabla(delta)
        return delta

    def update_nabla(self, delta):
        self.nabla_b = np.sum(delta, axis=1)
        self.nabla_w = np.dot(delta, self.before_a.transpose())
