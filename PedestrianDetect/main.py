import os
import cv2 as cv

path = r"E:\dataset\INRIAPerson\Train\pos"
for root, dirs, files in os.walk(path):
    for file in files:
        img = cv.imread(os.path.join(root, file))
        hog = cv.HOGDescriptor()
        hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
        (rects, weights) = hog.detectMultiScale(img,
                                                winStride=(4, 4),
                                                padding=(8, 8),
                                                scale=1.25,
                                                useMeanshiftGrouping=False)
        for (x, y, w, h) in rects:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv.imshow("hog-people", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
