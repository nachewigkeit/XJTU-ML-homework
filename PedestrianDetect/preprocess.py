import os
import random
import cv2 as cv
from PIL import Image
import numpy as np
from tqdm import tqdm


def getPosImages(path, border):
    hog = cv.HOGDescriptor()
    feature = []
    for root, dirs, files in os.walk(path):
        for file in tqdm(files):
            img = Image.open(os.path.join(root, file))
            img = img.convert("RGB")
            img = np.array(img)[border:border + 128, border:border + 64, ::-1]
            feature.append(hog.compute(img))

    random.shuffle(feature)
    return np.hstack(feature).T


def getNegImages(path, multiple=1):
    hog = cv.HOGDescriptor()
    feature = []
    for root, dirs, files in os.walk(path):
        for file in tqdm(files):
            img = Image.open(os.path.join(root, file))
            img = img.convert("RGB")
            img = np.array(img)

            for i in range(multiple):
                rand0 = random.randint(0, img.shape[0] - 128)
                rand1 = random.randint(0, img.shape[1] - 64)
                img = img[rand0:rand0 + 128, rand1:rand1 + 64, ::-1]
                feature.append(hog.compute(img))

    random.shuffle(feature)
    return np.hstack(feature).T


if __name__ == "__main__":
    path = r"E:\AI\dataset\INRIAPerson"

    feature = getPosImages(os.path.join(path, r"96X160H96\Train\pos"), 16)
    print("训练集正例数量：", feature.shape[0])
    np.save("data/trainPos.npy", feature)

    feature = getPosImages(os.path.join(path, r"70X134H96\Test\pos"), 3)
    print("测试集正例数量：", feature.shape[0])
    np.save("data/testPos.npy", feature)

    feature = getNegImages(os.path.join(path, r"Train\neg"), 2)
    print("训练集负例数量：", feature.shape[0])
    np.save("data/trainNeg.npy", feature)

    feature = getNegImages(os.path.join(path, r"Test\neg"), 2)
    print("测试集负例数量：", feature.shape[0])
    np.save("data/testNeg.npy", feature)
