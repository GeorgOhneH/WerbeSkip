import cupy
import cupyx


def ascupy(a):
    return cupy.array(a)


def scatter_add(a, indices, b=None):
    return cupyx.scatter_add(a, indices, b)
