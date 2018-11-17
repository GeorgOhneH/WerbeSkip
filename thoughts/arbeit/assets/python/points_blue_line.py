import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt

array = np.array(dd.io.load("points.h5"))
filters = []
result = []

filter_size = 25
chain_size = 5

for i in range(len(array)):
    snippet = array[i-filter_size:i]
    if np.any(snippet > 0.90):  # checks if network is sure that it found a logo
        filters.append(1)
    else:
        filters.append(0)

    last_filter = filters[-1]
    if np.all(np.array(filters[-chain_size:]) == last_filter):  # checks if the last values are the same
        if filters[-1] == 1:
            if np.mean(array[i-chain_size:i]) > 0.9:
                result.append(last_filter)
            else:
                result.append(result[-1])
        else:
            result.append(last_filter)
    else:
        result.append(result[-1])

result[:100] = [1] * 100

plt.figure(figsize=(15,5))
plt.plot(list(range(len(array))), array, "ro", ms=0.4)
plt.plot(list(range(len(result))), result, "b", ms=0.05, alpha=0.6)

plt.xlabel("Zeit")
plt.ylabel("Vorhersage")

plt.ioff()
plt.show()
