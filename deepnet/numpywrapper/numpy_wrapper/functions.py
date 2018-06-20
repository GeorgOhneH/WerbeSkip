import numpy


def asnumpy(a, stream=None):
    return numpy.asarray(a)


def ascupy(a):
    return a


def scatter_add(a, indices, b=None):
    numpy.add.at(a, indices, b)