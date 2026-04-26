import cv2
import numpy as np
from matplotlib import pyplot as plt

def lbp(img_gray):
    
    rows, cols = img_gray.shape
    lbp = np.zeros((rows, cols), dtype=np.uint8)


    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    powers = [1, 2, 4, 8, 16, 32, 64, 128]

    for (dr, dc), power in zip(offsets, powers):
        shifted = np.roll(np.roll(img_gray, -dr, axis=0), -dc, axis=1)
        lbp += ((shifted >= img_gray) * power).astype(np.uint8)

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


