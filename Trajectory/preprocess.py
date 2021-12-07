import numpy as np
import utils
from time import time

start = time()
data = utils.getData()
print(time() - start)
start = time()
d = utils.getDistMat(data)
print(time() - start)
np.save(r"data/data.npy", d)
