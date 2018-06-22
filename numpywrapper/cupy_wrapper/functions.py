import cupyx


def scatter_add(a, indices, b=None):
    return cupyx.scatter_add(a, indices, b)
