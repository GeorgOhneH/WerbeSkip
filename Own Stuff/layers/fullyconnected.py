import numpy as np


class FullyConnectedLayer(object):
    def __init__(self, neurons, activation):
        self.neurons = neurons
        self.activation = activation
        self.biases = None
        self.weights = None
        self.nabla_b = None
        self.nabla_w = None
        self.z = None
        self.before_a = None
        self.a = None

    def init_weights(self, neurons_before):
        self.biases = np.random.randn(self.neurons, 1)
        self.weights = np.random.randn(self.neurons, neurons_before) / np.sqrt(neurons_before)

    def reset_nabla(self):
        self.nabla_b = np.zeros(self.biases.shape)
        self.nabla_w = np.zeros(self.weights.shape)

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

    def calculate_loss(self, cost, y):
        loss = cost.function(self.a, y)
        return loss

    def make_first_delta(self, cost, y):
        delta = np.multiply(cost.delta(self.a, y), self.activation.derivative(self.z))
        self.update_nabla(delta)
        return delta

    def make_next_delta(self, delta, last_weights):
        delta = np.multiply(np.dot(last_weights.transpose(), delta), self.activation.derivative(self.z))
        self.update_nabla(delta)
        return delta

    def update_nabla(self, delta):
        delta = np.sum(delta, axis=1)
        self.nabla_b += delta
        self.nabla_w += np.dot(delta, self.before_a.transpose())
