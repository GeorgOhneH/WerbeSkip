from layers.layer import Layer

import numpy as np


class FullyConnectedLayer(Layer):
    """
    The FullyConnectedLayer saves the weights and biases of the layer
    As the name says it is fully connected, this means that every neuron
    from the previous layer is connect with every neuron of this layer

    The biases are from the neurons, which are after the weight connected
    """
    def __init__(self, neurons):
        """
        :param neurons: unsigned int
        """
        self.neurons = neurons
        self.optimizer = None
        self.biases = None
        self.weights = None
        self.nabla_b = None
        self.nabla_w = None
        self.a = None

    def __str__(self):
        return "{}: neurons: {}".format(super(FullyConnectedLayer, self).__str__(), self.neurons)

    def init(self, neurons_before, optimizer):
        """
        Initial the layer
        Uses Gaussian random variables with a mean of 0 and a standard deviation of 1
        :param neurons_before: unsigned int
        :param optimizer: optimiser of the Optimizer class
        :return: neurons of layer: unsigned int
        """
        self.biases = np.random.randn(self.neurons, 1)
        self.weights = np.random.randn(self.neurons, neurons_before) / np.sqrt(neurons_before)
        self.optimizer = optimizer
        return self.neurons

    def forward(self, a):
        """Applies a matrix multiplication of the weights and adds the biases """
        return self.weights @ a + self.biases

    def forward_backpropagation(self, a):
        """
        Applies a matrix multiplication of the weights and adds the biases
        and saves the value for the backpropagation
        """
        self.a = a
        return self.forward(a)

    def make_delta(self, delta):
        """Calculates error and the derivative of the parameters"""
        self.nabla_b = np.sum(delta, axis=1, keepdims=True)
        self.nabla_w = delta @ self.a.T
        return self.weights.T @ delta

    def adjust_parameters(self, mini_batch_size):
        """Changes the weights and biases after the optimizer calculates the change"""
        change_w, change_b = self.optimizer.calculate_change(self.nabla_w, self.nabla_b)
        self.weights -= change_w/mini_batch_size
        self.biases -= change_b/mini_batch_size
