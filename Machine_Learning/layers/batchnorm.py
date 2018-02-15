from layers.layer import Layer

import numpy as np


class BatchNorm(Layer):
    """
    BatchNorm is layer, which normalises the values

    It normalises the values, so that the have a standard derivative of 1
    and a mean of 0
    exactly like the weights and biases are initialized

    It has 2 learnable parameters, so the network can decide if needs normalisation
    or if it better works with out it

    BatchNorm helps  preventing overfitting and alloys higher learning rates
    """
    def __init__(self):
        """Takes no arguments"""
        self.mu_avg = 0
        self.var_avg = 1
        self.beta = None
        self.gamma = None
        self.nabla_g = None
        self.nabla_b = None
        self.optimizer = None
        self.a = None
        self.mu = None
        self.var = None
        self.norm = None

    def init(self, neurons_before, optimizer):
        """Initial gamma and beta, so that they don't have an effect"""
        self.optimizer = optimizer
        self.gamma = np.ones((neurons_before, 1))
        self.beta = np.zeros((neurons_before, 1))
        return neurons_before

    def forward(self, a):
        """
        Normalises the values
        with the average of the mean of the variance while training
        because the network uses SGD
        """
        norm = (a - self.mu_avg)/np.sqrt(self.var_avg + 1e-8)
        return np.multiply(self.gamma, norm) + self.beta

    def forward_backpropagation(self, a):
        """Normalises the values and saves the averages"""
        self.a = a
        self.mu = np.mean(a, axis=1, keepdims=True)
        self.var = np.var(a, axis=1, keepdims=True)
        self.norm = (a - self.mu)/np.sqrt(self.var + 1e-8)

        self.mu_avg = 0.9*self.mu_avg + 0.1*self.mu
        self.var_avg = 0.9*self.var_avg + 0.1*self.var

        return self.gamma * self.norm + self.beta

    def make_delta(self, delta):
        """Calculates the derivative and saves the error """
        self.nabla_g = np.sum(delta * self.norm, axis=1, keepdims=True)
        self.nabla_b = np.sum(delta, axis=1, keepdims=True)

        size = self.a.shape[1]

        a_mu = self.a - self.mu
        inv_var = 1 / np.sqrt(self.var + 1e-8)

        delta_norm = delta * self.gamma
        delta_var = np.sum(delta_norm * a_mu, axis=1, keepdims=True) * -0.5 * inv_var ** 3
        delta_mu = np.sum(delta_norm * -inv_var, axis=1, keepdims=True) + delta_var * np.mean(-2 * a_mu, axis=1, keepdims=True)

        delta = (delta_norm * inv_var) + (delta_var * 2 * a_mu / size) + (delta_mu / size)

        return delta

    def adjust_weights(self, mini_batch_size):
        """Adjust the param gamma and beta with the optimizer"""
        change_g, change_b = self.optimizer.calculate_change(self.nabla_g, self.nabla_b)
        self.gamma -= change_g/mini_batch_size
        self.beta -= change_b/mini_batch_size
