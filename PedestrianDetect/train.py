import cv2 as cv
import numpy as np
from sklearn import svm

posX = np.load(r"data/trainPos.npy")
posX = np.insert(posX, 0, values=1, axis=1)
negX = np.load(r"data/trainNeg.npy")
negX = np.insert(negX, 0, values=0, axis=1)
x = np.vstack((posX, negX))
np.random.shuffle(x)
y = x[:, 0].astype('int')
x = x[:, 1:]

'''
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svm.train(x, cv.ml.ROW_SAMPLE, y)
svm.save(r"weight/svm.xml")

yPred = svm.predict(x)
print(np.mean(y == yPred[1]))
'''

clf = svm.SVC()
clf.fit(x, y)

yPred = clf.predict(x)
print(np.mean(yPred == y))
