import numpy as np

x = np.arange(0, 9)
x.shape = (3, 3)
print(x[..., [0, 1]])
print(x.dtype)

