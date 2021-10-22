import os
import numpy as np
import cv2 as cv
from sklearn.externals import joblib
from tqdm import tqdm
from time import time


def detectMultiScale(img, clf, thresh, winStride, scale):
    hog = cv.HOGDescriptor()
    answer = [[], []]

    for i in tqdm(range(0, img.shape[0] - 128, winStride)):
        for j in range(0, img.shape[1] - 64, winStride):
            imgPart = img[i:i + 128, j:j + 64, :]
            feature = hog.compute(imgPart)
            answer[0].append(feature)
            answer[1].append((i, j, 128, 64))

    answer[0] = np.hstack(answer[0]).T
    answer[1] = np.array(answer[1])
    answer[0] = clf.predict_proba(answer[0])[:, 1]

    return answer[1][answer[0] > thresh]


clf = joblib.load(r"weight/svm")

path = r"E:\AI\dataset\INRIAPerson\Train\pos"
for root, dirs, files in os.walk(path):
    for file in files:
        img = cv.imread(os.path.join(root, file))
        rects = detectMultiScale(img, clf, thresh=0.9, winStride=20, scale=1.25)
        for (y, x, h, w) in rects:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv.imshow("hog-people", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
