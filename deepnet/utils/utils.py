import numpywrapper as np


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


def make_batches(a, batch_size):
    if batch_size >= a.shape[0]:
        return [a]

    batches = []
    for i in range(0, a.shape[0], batch_size):
        batches.append(a[i:batch_size + i])
    return batches


def flatten(a):
    mini_batch_size = a.shape[0]
    out = a.ravel().reshape((mini_batch_size, -1))
    return out


def unflatten(a, shape):
    out = a.reshape(shape)
    return out
