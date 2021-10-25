import numpy as np


def getData(posPath, negPath):
    posX = np.load(posPath)
    posX = np.insert(posX, 0, values=1, axis=1)
    negX = np.load(negPath)
    negX = np.insert(negX, 0, values=0, axis=1)
    x = np.vstack((posX, negX))
    np.random.shuffle(x)
    y = x[:, 0].astype('int')
    x = x[:, 1:]

    return x, y
