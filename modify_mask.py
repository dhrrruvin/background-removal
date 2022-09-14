import cv2 as cv
import numpy as np
import os

files = os.listdir("dataset/train/aug")

for i, file in enumerate(files):

    img = cv.imread(os.path.join("dataset/train/aug", file), 0)

    w = np.where(
        img[:, :] == 255
    )
    b = np.where(
        img[:, :] == 0
    )

    img[w] = 0
    img[b] = 1

    edge = cv.GaussianBlur(img, (3, 3), 0)

    edge = cv.Canny(edge, 100, 200)
    w = np.where(
        edge[:, :] == 255
    )
    edge[w] = 1
    nimg = cv.addWeighted(img, 1, edge, 1, 0)

    # cv.imshow("img", edge)
    # cv.imshow("img1", img)
    cv.imwrite(os.path.join("dataset/train/aug", file), nimg)