import numpy
import cupy


def asnumpy(a, stream=None):
    return numpy.asarray(a)


def ascupy(a):
    return cupy.asarray(a)


def asarray(a, dtype=None):
    return cupy.asnumpy(a).astype(dtype=dtype)


def scatter_add(a, indices, b=None):
    numpy.add.at(a, indices, b)
