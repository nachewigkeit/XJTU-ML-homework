import utils
import numpy as np
import matplotlib.pyplot as plt

time, id, pos = utils.getData()
print(time.min(), time.max())
print(id.min(), id.max())
print(pos[:, 0].min(), pos[:, 0].max())
print(pos[:, 1].min(), pos[:, 1].max())

length = []
for i in range(1, id.max() + 1):
    length.append(len(time[time == i]))
print(np.min(length), np.max(length), np.mean(length))
plt.hist(length)
plt.savefig("image/time.png", bbox_inches="tight")

dist = np.load("data/data.npy")
plt.hist(dist[dist < 100])
plt.savefig("image/dist.png", bbox_inches="tight")
