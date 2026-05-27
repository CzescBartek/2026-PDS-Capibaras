import cv2
import numpy as np
from math import nan
from skimage import color, exposure
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.transform import resize
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from statistics import variance, stdev
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import circmean, circvar, circstd

data_path = '../data/'

def load_image_and_mask(image_id, data_path=data_path):
    '''
    Docstring for load_image
    
    :param image_id: "img_id" from metadata.csv
    :param data_path: Relative path of the data folder

    This functions takes as input an image ID, 
    and returns the corresponding image and mask 
    (found in "/data/imgs/" and "/data/masks/" respectively)
    as an array
    '''
    img_path = data_path + "imgs/"
    mask_path = data_path + "masks/"

    file_im = img_path + image_id
    file_mask = (mask_path + image_id).replace(".png", "_mask.png")

    im = plt.imread(file_im)
    mask = plt.imread(file_mask)

    if im.shape[-1] == 4:
        im = im[:, :, :3]

    if len(mask.shape) == 3:
        mask = rgb2gray(mask)

    im = resize(im, (256, 256), anti_aliasing=True)
    mask = resize(mask, (256, 256), anti_aliasing=True)

    mask = mask > 0.5

    return im, mask


def slic_segmentation(image, mask, n_segments = 50, compactness = 0.1):
    '''Get color segments of lesion from SLIC algorithm.
    Optional argument n_segments (defualt 50) defines desired amount of segments.
    Optional argument compactness (defualt 0.1) defines balance between color
    and position.

    Args:
        image (numpy.ndarray): image to segment
        mask (numpy.ndarray):  image mask
        n_segments (int, optional): desired amount of segments
        compactness (float, optional): compactness score, decides balance between
            color and and position

    Returns:
        slic_segments (numpy.ndarray): SLIC color segments.
    '''
    slic_segments = slic(image,
                    n_segments = n_segments,
                    compactness = compactness,
                    sigma = 1,
                    mask = mask,
                    start_label = 1,
                    channel_axis = 2)

    return slic_segments


def get_rgb_means(image, mask):
    '''Get mean RGB values for each segment in a SLIC segmented image.

    Args:
        image (numpy.ndarray): original image
        slic_segments (numpy.ndarray): SLIC segmentation

    Returns:
        hsv_means (list): HSV mean values for each segment.
    '''

    if image.size == 0 or np.sum(mask) == 0:
        return None
    slic_segments = slic_segmentation(image, mask)

    max_segment_id = np.unique(slic_segments)[-1]

    hsv_means = []
    for i in range(1, max_segment_id + 1):

        #Create masked image where only specific segment is active
        segment = image.copy().astype(np.int16)
        segment[slic_segments != i] = -1

        #Get average RGB values from segment
        rgb_mean = np.mean(segment, axis = (0, 1), where = (segment != -1))

        rgb_means.append(rgb_mean)

    return rgb_means

