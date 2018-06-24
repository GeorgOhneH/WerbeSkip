import numpy


def asnumpy(a, stream=None):  # steam for compatibility with cupy's version
    return numpy.asarray(a)


def scatter_add(a, indices, b=None):
    numpy.add.at(a, indices, b)
