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

def get_hsv_means(image, mask):
    '''Get mean HSV values for each segment in a SLIC segmented image.

    Args:
        image (numpy.ndarray): original image
        slic_segments (numpy.ndarray): SLIC segmentation

    Returns:
        hsv_means (list): HSV mean values for each segment.
    '''
    slic_segments = slic_segmentation(image, mask)
    hsv_image = rgb2hsv(image)

    max_segment_id = np.unique(slic_segments)[-1]

    hsv_means = []
    for i in range(1, max_segment_id + 1):

        # Create masked image where only specific segment is active
        segment = hsv_image.copy()
        segment[slic_segments != i] = nan

        #Get average HSV values from segment
        hue_mean = circmean(segment[:, :, 0], high=1, low=0, nan_policy='omit') # Compute circular hue mean
        sat_mean = np.mean(segment[:, :, 1], where = (slic_segments == i)) # Compute saturation mean
        val_mean = np.mean(segment[:, :, 2], where = (slic_segments == i)) # Compute value mean

        hsv_mean = np.asarray([hue_mean, sat_mean, val_mean])

        hsv_means.append(hsv_mean)

    return hsv_means

def hsv_var(image, mask):
    '''Get variance of HSV means for each segment in
    SLIC segmentation in hue, saturation and value channels

    Args:
        image (numpy.ndarray): image to compute color variance for
        slic_segments (numpy.ndarray): array containing SLIC segmentation

    Returns:
        hue_var (float): variance in hue channel segment means
        sat_var (float): variance in saturation channel segment means
        val_var (float): variance in value channel segment means.
    '''
    if np.sum(mask) == 0:
        return np.nan, np.nan, np.nan

    slic_segments = slic_segmentation(image,mask)
    # If there is only 1 slic segment, return (0, 0, 0)
    if len(np.unique(slic_segments)) == 2: # Use 2 since slic_segments also has 0 marking for area outside mask
        return 0, 0, 0

    hsv_means = get_hsv_means(image, slic_segments)
    n = len(hsv_means) # Amount of segments, used later to compute variance

    # Seperate and collect channel means together in lists
    hue = []
    sat = []
    val = []
    for hsv_mean in hsv_means:
        hue.append(hsv_mean[0])
        sat.append(hsv_mean[1])
        val.append(hsv_mean[2])

    # Compute variance for each channel seperately
    hue_var = circvar(hue, high=1, low=0)
    sat_var = variance(sat, sum(sat)/n)
    val_var = variance(val, sum(val)/n)


    return hue_var, sat_var, val_var