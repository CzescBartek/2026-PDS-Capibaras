import cv2
import numpy as np
from math import sqrt, floor, ceil, nan, pi
from skimage import color, exposure
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.transform import rotate
from skimage import morphology
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from scipy.stats import circmean, circvar, circstd
from statistics import variance, stdev
from scipy.spatial import ConvexHull

def cut_mask(mask):
    '''Cut empty space from mask array such that it has smallest possible dimensions.

    Args:
        mask (numpy.ndarray): mask to cut

    Returns:
        cut_mask_ (numpy.ndarray): cut mask
    '''
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = [index for index, col_sum in enumerate(col_sums) if col_sum != 0]
    active_rows = [index for index, row_sum in enumerate(row_sums) if row_sum != 0]

    col_min, col_max = active_cols[0], active_cols[-1]
    row_min, row_max = active_rows[0], active_rows[-1]

    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1]

    return cut_mask_


def asymmetry(mask):
    '''Calculate asymmetry score between 0 and 1 from vertical and horizontal axis.
    0 = complete symmetry, 1 = complete asymmetry.

    Args:
        mask (numpy.ndarray): input mask

    Returns:
        asymmetry_score (float): Float between 0 and 1 indicating level of asymmetry.
    '''
    row_mid = mask.shape[0] / 2
    col_mid = mask.shape[1] / 2

    upper_half = mask[:ceil(row_mid), :]
    lower_half = mask[floor(row_mid):, :]
    left_half  = mask[:, :ceil(col_mid)]
    right_half = mask[:, floor(col_mid):]

    flipped_lower = np.flip(lower_half, axis=0)
    flipped_right = np.flip(right_half, axis=1)

    hori_xor_area = np.logical_xor(upper_half, flipped_lower)
    vert_xor_area = np.logical_xor(left_half, flipped_right)

    total_pxls         = np.sum(mask)
    hori_asymmetry_pxls = np.sum(hori_xor_area)
    vert_asymmetry_pxls = np.sum(vert_xor_area)

    asymmetry_score = (hori_asymmetry_pxls + vert_asymmetry_pxls) / (total_pxls * 2)

    return round(asymmetry_score, 4)

def compactness_score(mask):
    '''Computes a compactness score for the given mask.
    The score is based of the Polsby-Popper measure.
    The score falls between the value 0 and 1. Scores closer to 1 indicates a more compact mask.

    Args:
        mask (numpy.ndarray): input masked image

    Returns:
        compactness_score (float): Float between 0 and 1 indicating compactness.
    '''

     #Area of ground truth
    A = np.sum(mask)

    #Structural element, that we will use as a "brush" on our mask
    struct_el = morphology.disk(2)

    # Use this "brush" to erode the image - eat away at the borders
    mask_eroded = morphology.binary_erosion(mask, struct_el)

    #Finding the perimeter of the ground truth
    perimeter = mask - mask_eroded

    #Length of the perimeter
    l = np.sum(perimeter)

    compactness = (4*pi*A)/(l**2)

    score = round(1-compactness, 3)

    return score

def convexity_score(mask):
    '''Calculate convexity score between 0 and 1,
    with 0 indicating a smoother border and 1 a more crooked border.

    Args:
        image (numpy.ndarray): input masked image

    Returns:
        convexity_score (float): Float between 0 and 1 indicating convexity.
    '''

    # Get coordinates of all pixels in the lesion mask
    coords = np.transpose(np.nonzero(mask))

    # Compute convex hull of lesion pixels
    hull = ConvexHull(coords)

    # Compute area of lesion mask
    lesion_area = np.count_nonzero(mask)

    # Compute area of convex hull
    convex_hull_area = hull.volume + hull.area

    # Compute convexity as ratio of lesion area to convex hull
    convexity = lesion_area / convex_hull_area

    return convexity #round(1-convexity, 3)

def get_compactness(mask):
    # mask = color.rgb2gray(mask)
    area = np.sum(mask)

    struct_el = morphology.disk(3)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    perimeter = np.sum(mask - mask_eroded)

    return perimeter**2 / (4 * np.pi * area)

def mean_asymmetry(mask, rotations = 30):
    '''Return mean asymmetry score from mask.
    Optional argument (defualt 30) rotations decides amount of rotations in asymmetry calculation

    Args:
        mask (numpy.ndarray): mask to compute asymmetry score for
        rotations (int, optional): amount of rotations (default 30)

    Returns:
        mean_score (float): mean asymmetry score.
    '''
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    mean_score = sum(asymmetry_scores.values()) / len(asymmetry_scores)

    return mean_score

def rotation_asymmetry(mask, n: int):
    '''Rotate mask n times and calculate asymmetry score for each iteration.
    Rotates n times between 0 and 90 degrees, as 90 degree rotations do not change the
    asymmetry score, i.e., a 30 degree rotation is the same as a 120 degree rotation.

    Args:
        mask (numpy.ndarray): input mask
        n (int): amount of rotations

    Returns:
        asymmetry_scores (dict): dict of asymmetry scores calculated from each rotation.
    '''
    asymmetry_scores = {}

    for i in range(n):

        degrees = 90 * i / n

        rotated_mask = rotate(mask, degrees)
        cutted_mask = cut_mask(rotated_mask)

        asymmetry_scores[degrees] = asymmetry(cutted_mask)

    return asymmetry_scores