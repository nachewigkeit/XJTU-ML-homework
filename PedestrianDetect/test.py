import numpy as np
import cv2 as cv
import joblib
import matplotlib.pyplot as plt
from time import time


def detectMultiScale(img, clf, thres, winStride, scale):
    start = time()
    hog = cv.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    store = [[], []]

    scaledImg = img
    while scaledImg.shape[0] > 128 and scaledImg.shape[1] > 64:
        scaledImg = cv.resize(scaledImg, (0, 0), fx=1 / scale, fy=1 / scale)
        rate = (img.shape[0] / scaledImg.shape[0], img.shape[1] / scaledImg.shape[1])
        for i in range(0, scaledImg.shape[0] - 128, winStride):
            for j in range(0, scaledImg.shape[1] - 64, winStride):
                imgPart = scaledImg[i:i + 128, j:j + 64, :]
                feature = hog.compute(imgPart)
                store[0].append(feature)
                store[1].append((rate[0] * i, rate[1] * j, rate[0] * (i + 128), rate[1] * (j + 64)))

    store[0] = np.hstack(store[0]).T
    store[1] = np.array(store[1]).astype('int')
    conf = clf.predict_proba(store[0])[:, 1]
    print(time() - start)

    return store[1][conf > thres], conf[conf > thres]


def NMS(dets, scores, thres):
    # x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order是按照score降序排序的，得到的是排序的本来的索引，不是排完序的原数组
    order = scores.argsort()[::-1]
    # ::-1表示逆序

    temp = []
    while order.size > 0:
        i = order[0]
        temp.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标
        # 由于numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，需要用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thres)[0]
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return temp


def pipeline(clf, clsThres, iouThres):
    file = r"E:\AI\dataset\INRIAPerson\Train\pos\crop001001.png"
    img = cv.imread(file)
    rects, conf = detectMultiScale(img, clf, clsThres, winStride=4, scale=1.2)
    pos = NMS(rects, conf, iouThres)
    for (y1, x1, y2, x2) in rects[pos, :]:
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img[:, :, ::-1]


clf = joblib.load(r"weight/svm2")
clsThreshes = [0.5, 0.9, 0.99]
iouThreshes = [0.2, 0.4, 0.6]
images = []
for clsThres in clsThreshes:
    for iouThres in iouThreshes:
        print("clsThres", clsThres, "IoUThres:", iouThres)
        images.append(pipeline(clf, clsThres, iouThres))

clsLen = len(clsThreshes)
iouLen = len(iouThreshes)
plt.figure(figsize=(10 * iouLen, 10 * clsLen))
for i in range(clsLen):
    for j in range(iouLen):
        plt.subplot(clsLen, iouLen, i * iouLen + j + 1)
        plt.title("clsThres:" + str(clsThreshes[i]) + ",iouThres:" + str(iouThreshes[j]))
        plt.xticks([])
        plt.xticks([])
        plt.imshow(images[i])
plt.show()
