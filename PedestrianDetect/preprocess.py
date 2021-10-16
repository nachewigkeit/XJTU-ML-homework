import cv2 as cv
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import time


def getPosImages(path, border):
    hog = cv.HOGDescriptor()
    feature = []
    for root, dirs, files in os.walk(path):
        for file in tqdm(files):
            img = Image.open(os.path.join(root, file))
            img = img.convert("RGB")
            img = np.array(img)[border:border + 128, border:border + 64, ::-1]
            feature.append(hog.compute(img))

    return np.hstack(feature).T


if __name__ == "__main__":
    feature = getPosImages(r"E:\dataset\INRIAPerson\96X160H96\Train\pos", 16)
    print(feature.shape)
    np.save("data/trainPos.npy", feature)

    feature = getPosImages(r"E:\dataset\INRIAPerson\70X134H96\Test\pos", 3)
    print(feature.shape)
    np.save("data/testPos.npy", feature)
