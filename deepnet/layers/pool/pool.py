from layers.layer import Layer
from utils.im2col import *

import numpywrapper as np


class PoolLayer(Layer):
    """
    """

    def __init__(self, width_filter, height_filter, stride):
        """
        None functional
        """
        self.width_filter = width_filter
        self.height_filter = height_filter
        self.stride = stride
        self.a = None
        self.a_col = None

    def __str__(self):
        return "{}:  width_filter: {} height_filter: {} stride: {}" \
            .format(super(PoolLayer, self).__str__(), self.width_filter, self.height_filter, self.stride)

    def init(self, neurons_before, optimizer):
        depth, height, width = neurons_before
        height_out = (height - self.height_filter) / self.stride + 1
        width_out = (width - self.width_filter) / self.stride + 1

        if not width_out.is_integer() or not height_out.is_integer():
            raise ValueError("Doesn't work with theses Values")

        return depth, int(height_out), int(width_out)

    def forward(self, a):
        mini_batch_size, depth, height, width = a.shape
        height_out = (height - self.height_filter) / self.stride + 1
        width_out = (width - self.width_filter) / self.stride + 1

        if not width_out.is_integer() or not height_out.is_integer():
            raise ValueError("Doesn't work with theses Values")

        height_out, width_out = int(height_out), int(width_out)

        a_reshaped = a.reshape(mini_batch_size * depth, 1, height, width)
        self.a_col = im2col_indices(a_reshaped, self.height_filter, self.width_filter, padding=0, stride=self.stride)

        out = self.pool(self.a_col)

        out = out.reshape(height_out, width_out, mini_batch_size, depth)
        out = out.transpose(2, 3, 0, 1)

        return out

    def forward_backpropagation(self, a):
        self.a = a
        return self.forward(a)

    def pool(self, a_col):
        raise NotImplemented

    def make_delta(self, delta):
        """Calculates error and the derivative of the parameters"""
        mini_batch_size, depth, height, width = self.a.shape

        delta_a_col = np.zeros_like(self.a_col, dtype="float32")
        delta_out_col = delta.transpose(2, 3, 0, 1).ravel()

        delta_a_col = self.pool_delta(delta_a_col, delta_out_col)

        delta_out = col2im_indices(delta_a_col, (mini_batch_size * depth, 1, height, width),
                                   self.height_filter, self.width_filter, padding=0, stride=self.stride)

        delta_out = delta_out.reshape(self.a.shape)
        return delta_out

    def pool_delta(self, delta_a_col, delta_out_col):
        raise NotImplemented
