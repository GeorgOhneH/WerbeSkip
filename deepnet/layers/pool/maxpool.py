from layers.pool.pool import PoolLayer

import cupy as np


class MaxPoolLayer(PoolLayer):
    """
    """
    def __init__(self, width_filter, height_filter, stride):
        """
        """
        super().__init__(width_filter, height_filter, stride)
        self.max_indexes = None

    def __str__(self):
        return "{}:  width_filter: {} height_filter: {} stride: {}"\
                .format(super(MaxPoolLayer, self).__str__(), self.width_filter, self.height_filter, self.stride)

    def pool(self, a_col):
        self.max_indexes = np.argmax(a_col, axis=0)
        out = a_col[self.max_indexes, list(range(self.max_indexes.size))]
        return out

    def pool_delta(self, delta_a_col, delta_out_col):
        delta_a_col[self.max_indexes, list(range(self.max_indexes.size))] = delta_out_col
        return delta_a_col
