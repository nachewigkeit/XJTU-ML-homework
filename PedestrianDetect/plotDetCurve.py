from sklearn.metrics import plot_det_curve, plot_roc_curve
import joblib
from utils import getData
import matplotlib.pyplot as plt

negRate = 2
kers = ['linear', 'poly', 'rbf', 'sigmoid']
cs = [0.1, 1, 10]
gammas = [3e-4, 3e-3, 3e-2, 3e-1]

testX, testY = getData("data/" + str(negRate) + "/testPos.npy", "data/" + str(negRate) + "/testNeg.npy")

ax = plt.gca()
'''
plt.title('kernel')
for ker in kers:
    clf = joblib.load(r'weight/svm' + str(negRate) + ker)
    plot_det_curve(clf, testX, testY, name=ker, ax=ax)
plt.savefig(r"image/kernel.png")
plt.show()
'''

plt.title('C')
for ker in ['rbf']:
    for c in cs:
        clf = joblib.load(r'weight/svm' + str(negRate) + ker + 'c_' + str(c))
        plot_det_curve(clf, testX, testY, name=str(c), ax=ax)
plt.savefig(r"image/c.png")
plt.show()

'''
plt.title('gamma')
for ker in ['rbf']:
    for c in [1]:
        for gamma in gammas:
            clf = joblib.load(r'weight/svm' + str(negRate) + ker + 'c_' + str(c) + 'g_' + str(gamma))
            plot_det_curve(clf, testX, testY, name=str(gamma), ax=ax)
plt.savefig(r"image/gamma.png")
plt.show()
'''
