import csv
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
from scipy.spatial.distance import squareform


def getData():
    path = r"E:\AI\dataset\Trajectory\students003.txt"

    time = []
    id = []
    pos = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        for row in spamreader:
            time.append(int(float(row[0]) / 10))
            id.append(int(float(row[1])))
            pos.append([float(row[2]), float(row[3])])

    time = np.array(time)
    id = np.array(id)
    pos = np.array(pos)
    return time, id, pos


@jit(nopython=True)
def getDist(data, id1, id2):
    time, id, pos = data
    time1 = time[id == id1]
    time2 = time[id == id2]
    start = max(time1[0], time2[0])
    end = min(time1[-1], time2[-1])
    if end < start:
        return 100
    else:
        start2 = min(time1[0], time2[0])
        end2 = max(time1[-1], time2[-1])

        maximum = 0.
        for i in range(start2, end2 + 1):
            if i < time1[0]:
                v = pos[(time == time1[1]) & (id == id1)] - pos[(time == time1[0]) & (id == id1)]
                pos1 = pos[(time == time1[0]) & (id == id1)] + (i - time1[0]) * v
            elif i > time1[-1]:
                v = pos[(time == time1[-1]) & (id == id1)] - pos[(time == time1[-2]) & (id == id1)]
                pos1 = pos[(time == time1[-1]) & (id == id1)] + (i - time1[-1]) * v
            else:
                pos1 = pos[(time == i) & (id == id1)]

            if i < time2[0]:
                v = pos[(time == time2[1]) & (id == id2)] - pos[(time == time2[0]) & (id == id2)]
                pos2 = pos[(time == time2[0]) & (id == id2)] + (i - time2[0]) * v
            elif i > time2[-1]:
                v = pos[(time == time2[-1]) & (id == id2)] - pos[(time == time2[-2]) & (id == id2)]
                pos2 = pos[(time == time2[-1]) & (id == id2)] + (i - time2[-1]) * v
            else:
                pos2 = pos[(time == i) & (id == id2)]

            d = np.linalg.norm(pos1 - pos2)
            if d > maximum:
                maximum = d

        return maximum


def getDistMat(data):
    time, id, pos = data
    peopleNum = max(id)
    mat = np.zeros((peopleNum, peopleNum))

    for i in range(peopleNum):
        for j in range(i + 1, peopleNum):
            mat[i, j] = mat[j, i] = getDist(data, i + 1, j + 1)

    return squareform(mat, force="tovector")


def plotTrajectory(data, cluster, n):
    time, id, pos = data
    count = Counter(cluster)
    color = list(mcolors.TABLEAU_COLORS.keys())
    need = count.most_common(n)
    for i in range(len(need)):
        people = np.where(cluster == need[i][0])[0]
        for p in people:
            point = pos[id == p+1]
            plt.plot(point[:, 0], point[:, 1], c=color[i])
            step = 3
            plt.arrow(point[0, 0], point[0, 1], point[step, 0] - point[0, 0], point[step, 1] - point[0, 1],
                      head_width=0.2, zorder=5)


if __name__ == "__main__":
    data = getData()
    print(getDist(data, 26, 28))
