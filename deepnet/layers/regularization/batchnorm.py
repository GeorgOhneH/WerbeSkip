from deepnet.layers import Layer
from deepnet.utils import flatten, unflatten

import numpywrapper as np


class BatchNorm(Layer):
    """
    after https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
    BatchNorm is layer, which normalises the values

    It normalises the values, so that they have a standard derivative of 1
    and a mean of 0
    exactly like the weights and biases are normally initialized

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

        size = np.array(neurons_before)
        size = int(np.prod(size))

        self.gamma = np.ones((1, size), dtype="float32")
        self.beta = np.zeros((1, size), dtype="float32")
        return neurons_before

    def forward(self, a):
        """
        Normalises the values
        with the average of the mean of the variance while training
        because the network uses SGD
        """
        a_flatten = flatten(a)

        norm = (a_flatten - self.mu_avg)/np.sqrt(self.var_avg + 1e-8)
        out = self.gamma * norm + self.beta

        return unflatten(out, a.shape)

    def forward_backpropagation(self, a):
        """Normalises the values and saves the averages"""
        self.a = flatten(a)

        self.mu = np.mean(self.a, axis=0, keepdims=True)
        self.var = np.var(self.a, axis=0, keepdims=True)
        self.norm = (self.a - self.mu)/np.sqrt(self.var + 1e-8)

        # saving the average with decay
        self.mu_avg = 0.9*self.mu_avg + 0.1*self.mu
        self.var_avg = 0.9*self.var_avg + 0.1*self.var

        out = self.gamma * self.norm + self.beta

        return unflatten(out, a.shape)

    def make_delta(self, delta):
        """Calculates the derivative and saves the error """
        delta_shape = delta.shape
        delta = flatten(delta)
        self.nabla_g = np.sum(delta * self.norm, axis=0, keepdims=True)
        self.nabla_b = np.sum(delta, axis=0, keepdims=True)

        size = self.a.shape[0]

        a_mu = self.a - self.mu
        inv_var = 1 / np.sqrt(self.var + 1e-8)

        delta_norm = delta * self.gamma
        delta_var = np.sum(delta_norm * a_mu, axis=0, keepdims=True) * -0.5 * inv_var ** 3
        delta_mu = np.sum(delta_norm * -inv_var, axis=0, keepdims=True) + delta_var * np.mean(-2 * a_mu, axis=0, keepdims=True)

        delta = (delta_norm * inv_var) + (delta_var * 2 * a_mu / size) + (delta_mu / size)

        return unflatten(delta, delta_shape)

    def adjust_parameters(self, mini_batch_size):
        """Adjust the param gamma and beta with the optimizer"""
        change_g, change_b = self.optimizer.calculate_change(self.nabla_g, self.nabla_b)
        self.gamma -= change_g/mini_batch_size
        self.beta -= change_b/mini_batch_size

    def save(self):
        return [self.gamma, self.beta, self.mu_avg, self.var_avg]

    def load(self, array):
        self.gamma, self.beta, self.mu_avg, self.var_avg = array
