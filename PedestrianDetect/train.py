import numpy as np
from sklearn import svm
from sklearn.externals import joblib


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


negRate = 2
trainX, trainY = getData("data/" + str(negRate) + "/trainPos.npy", "data/" + str(negRate) + "/trainNeg.npy")
testX, testY = getData("data/" + str(negRate) + "/testPos.npy", "data/" + str(negRate) + "/testNeg.npy")

clf = svm.SVC(kernel="linear", probability=True)
clf.fit(trainX, trainY)
joblib.dump(clf, r'weight/svm' + str(negRate))

trainYPred = clf.predict(trainX)
print("测试集准确率：", np.mean(trainYPred == trainY))

testYPred = clf.predict(testX)
print("验证集准确率：", np.mean(testYPred == testY))
