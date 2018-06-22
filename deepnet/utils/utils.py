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


def cubify(arr, newshape):
    """https://stackoverflow.com/questions/42297115/numpy-split-cube-into-cubes/42298440#42298440"""
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)


def flatten(a):
    mini_batch_size = a.shape[0]
    out = a.ravel().reshape((mini_batch_size, -1))
    return out


def unflatten(a, shape):
    out = a.reshape(shape)
    return out
