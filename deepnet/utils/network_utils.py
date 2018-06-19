import cupy as np


def shuffle(x, y):
    """
    Shuffles data in unison with helping from indexing
    :param x: ndarray
    :param y: ndarray
    :return x, y: ndarray
    """
    indexes = np.random.permutation(x.shape[0])
    return x[indexes], y[indexes]


def make_mini_batches(x, y, size):
    """
    Shuffles data and makes mini batches
    :param x: ndarray
    :param y: ndarray
    :param size: unsigned int
    :return mini_batches: list with ndarray's
    """
    x, y = shuffle(x, y)
    mini_batches = []
    for i in range(0, x.shape[0], size):
        mini_batches.append((x[i:size + i], y[i:size + i]))
    return mini_batches


def blockshaped(arr, nrows, ncols):
    """
    https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    try:
        blocks = (arr.reshape(h // nrows, nrows, -1, ncols)
                  .swapaxes(1, 2)
                  .reshape(-1, nrows, ncols))
    except ValueError:
        blocks = []

    return blocks


def flatten(a):
    mini_batch_size = a.shape[0]
    out = a.ravel().reshape((mini_batch_size, -1))
    return out


def unflatten(a, shape):
    out = a.reshape(shape)
    return out
