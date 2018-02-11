import numpy as np


def shuffle(x, y):
    # shuffles data in unison with helping from indexing
    indexes = np.random.permutation(x.shape[1])
    return x[..., indexes], y[..., indexes]


def make_mini_batches(x, y, size):
    x, y = shuffle(x, y)
    mini_batches = []
    length = x.shape[1]
    for i in range(0, length, size):
        mini_batches.append((x[:, i:size + i], y[:, i:size + i]))
    return mini_batches
