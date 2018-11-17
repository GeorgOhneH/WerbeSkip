import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt

array = np.array(dd.io.load("points.h5"))

plt.figure(figsize=(15,5))
plt.plot(list(range(len(array))), array, "ro", ms=0.4)
# plt.plot(list(range(len(result))), result, "b", ms=0.05, alpha=0.5)

plt.xlabel("Zeit")
plt.ylabel("Vorhersage")

plt.ioff()
plt.show()
