import cv2
import numpy as np
from matplotlib import pyplot as plt
 
 
def fast_lbp_synchronized(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.int32)
    rows, cols = img_gray.shape
    img_lbp = np.zeros((rows, cols), dtype=np.uint8)


    offsets = [
        (-1, -1), ( -1, 0), (-1, 1), (0, 1), 
        ( 1,  1), (  1, 0), ( 1, -1), (0, -1)
    ]
    
    powers = [1, 2, 4, 8, 16, 32, 64, 128]


    for i in range(len(offsets)):
        dy, dx = offsets[i]
        pwr = powers[i]
        
        shifted = np.zeros_like(img_gray)
        
        r_start, r_end = max(0, dy), rows + min(0, dy)
        c_start, c_end = max(0, dx), cols + min(0, dx)
        
        img_r_start, img_r_end = max(0, -dy), rows + min(0, -dy)
        img_c_start, img_c_end = max(0, -dx), cols + min(0, -dx)
        
        compare = img_gray[img_r_start:img_r_end, img_c_start:img_c_end] >= img_gray[r_start:r_end, c_start:c_end]
        
        shifted[r_start:r_end, c_start:c_end] = compare * pwr
        img_lbp += shifted.astype(np.uint8)

    # Histogram
    hist, _ = np.histogram(img_lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist