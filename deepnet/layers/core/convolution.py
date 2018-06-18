from layers.layer import Layer
from utils.im2col import *

import numpy as np


class ConvolutionLayer(Layer):
    """
    """
    def __init__(self, n_filter, width_filter, height_filter, stride, zero_padding):
        """
        """
        self.n_filter = n_filter
        self.width_filter = width_filter
        self.height_filter = height_filter
        self.stride = stride
        self.zero_padding = zero_padding
        self.optimizer = None
        self.biases = None
        self.weights = None
        self.width_out = None
        self.height_out = None
        self.mini_batch_size = None
        self.width = None
        self.height = None
        self.depth = None
        self.a_col = None
        self.nabla_b = None
        self.nabla_w = None

    def __str__(self):
        return "{}: n_filter: {} width_filter: {} height_filter: {} stride: {} zero_padding: {}"\
                .format(super(ConvolutionLayer, self).__str__(), self.n_filter, self.width_filter,
                        self.height_filter, self.stride, self.zero_padding)

    def init(self, neurons_before, optimizer):
        """
        Initial the layer
        Uses Gaussian random variables with a mean of 0 and a standard deviation of 1
        :param neurons_before: tuple
        :param optimizer: optimiser of the Optimizer class
        :return: neurons of layer: unsigned int
        """
        self.width, self.height, self.depth = neurons_before
        self.biases = np.random.randn(self.n_filter, 1)
        self.weights = np.random.randn(self.n_filter, self.depth, self.height_filter, self.width_filter) / np.sqrt(neurons_before)
        self.optimizer = optimizer

        self.width_out = (self.width - self.width_filter + 2 * self.zero_padding) / self.stride + 1
        self.height_out = (self.height - self.height_filter + 2 * self.zero_padding) / self.stride + 1
        if not self.height_out.is_integer() or not self.width_out.is_integer():
            raise ValueError("Doesn't work with theses Values.")

        return int(self.width_out), int(self.height_out), self.n_filter

    def forward(self, a):
        self.mini_batch_size = a.shape[0]

        self.a_col = im2col_indices(a, self.height_filter, self.width_filter, stride=self.stride, padding=self.zero_padding)
        weights_row = self.weights.reshape(self.n_filter, -1)

        out = weights_row @ self.a_col + self.biases
        out = out.reshape(self.n_filter, self.height_out, self.width_out, self.mini_batch_size)
        out = out.transpose(3, 0, 1, 2)
        return out

    def forward_backpropagation(self, a):
        return self.forward(a)

    def make_delta(self, delta):
        """Calculates error and the derivative of the parameters"""
        delta_flat = delta.transpose(1, 2, 3, 0).reshape(self.n_filter, -1)

        self.nabla_w = delta_flat @ self.a_col.T
        self.nabla_w = self.nabla_w.reshape(self.weights.shape)

        self.nabla_b = np.sum(delta, axis=(0, 2, 3)).reshape(self.n_filter, -1)

        weights_flat = self.weights.reshape(self.n_filter, -1)

        delta_col = weights_flat.T @ delta_flat
        shape = (self.mini_batch_size, self.depth, self.height, self.width)
        delta = col2im_indices(delta_col, shape, self.height_filter, self.width_filter, self.zero_padding, self.stride)

        return delta

    def adjust_parameters(self, mini_batch_size):
        """Changes the weights and biases after the optimizer calculates the change"""
        change_w, change_b = self.optimizer.calculate_change(self.nabla_w, self.nabla_b)
        self.weights -= change_w/mini_batch_size
        self.biases -= change_b/mini_batch_size
