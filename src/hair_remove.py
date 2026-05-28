import cv2
import numpy as np
from hair_coverage import hair_coverage

def removeHair_auto(img_org, img_gray):


    coverage = hair_coverage(img_gray)

    if coverage <= 0.005:
        return img_org
    elif coverage < 0.035 and coverage > 0.005:
        kernel_size = (12, 12)
    elif coverage >= 0.035:
        kernel_size = (25,25)


    kernel = cv2.getStructuringElement(1, kernel_size)
    
    whitehat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    if np.sum(blackhat) > np.sum(whitehat):
        combined = blackhat
    else:
        combined = whitehat

    _, mask = cv2.threshold(combined, 10, 255, cv2.THRESH_BINARY)
    img_out = cv2.inpaint(img_org, mask, 1, cv2.INPAINT_TELEA)

    return img_out


