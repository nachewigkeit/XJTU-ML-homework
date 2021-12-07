import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, fcluster
import matplotlib.pyplot as plt
import utils

data = utils.getData()
dist = np.load("data/data.npy")
mergings = linkage(dist, method='complete')

thres = [3, 5, 7]
for t in thres:
    plt.figure()
    cluster = fcluster(mergings, t, criterion="distance")
    plt.title("Cluster Number:" + str(max(cluster)))
    plt.xlim([0, 15])
    plt.ylim([0, 13])
    utils.plotTrajectory(data, cluster, 3)
    plt.savefig("image/" + str(t) + ".png", bbox_inches="tight")
