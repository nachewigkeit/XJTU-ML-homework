import numpy as np
from sklearn import svm
import joblib
from utils import getData

negRate = 2
kers = ['rbf']
cs = [100, 1000]
gammas = ['scale']

trainX, trainY = getData("data/" + str(negRate) + "/trainPos.npy", "data/" + str(negRate) + "/trainNeg.npy")
testX, testY = getData("data/" + str(negRate) + "/testPos.npy", "data/" + str(negRate) + "/testNeg.npy")

for ker in kers:
    for c in cs:
        for gamma in gammas:
            print("kernel:", ker, ",c:", c, ",gamma:", gamma)
            clf = svm.SVC(kernel=ker, C=c, gamma=gamma, probability=True)
            clf.fit(trainX, trainY)
            joblib.dump(clf, r'weight/svm' + str(negRate) + ker + 'c_' + str(c))
            # joblib.dump(clf, r'weight/svm' + str(negRate) + ker + 'c_' + str(c) + 'g_' + str(gamma))

            trainYPred = clf.predict(trainX)
            print("训练集准确率：", np.mean(trainYPred == trainY))

            testYPred = clf.predict(testX)
            print("验证集准确率：", np.mean(testYPred == testY))
