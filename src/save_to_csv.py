import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor, pi
from FEATURE_A import asymmetry, midpointGroup9
import os
from FEATURE_COLOR import extract_features, get_relative_rgb_means,color_dominance,hsv_var,rgb_var,get_hsv_means,get_rgb_means,load_image_and_mask,cut_im_by_mask,slic_segmentation,get_com_col,get_multicolor_rate
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
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("../data/metadata.csv")
image_id = df["img_id"].tolist()

data_path = '../data/'
imgs_path = "../data/imgs/"
mask_path= "../data/masks/"

csv_path = '../data/features.csv'
# CHECKING IF THE FILE EXISTS OR IS IT EMPTY
if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:

    df_features = pd.read_csv(csv_path)
else:

    df_features = pd.DataFrame({'img_id': image_id})


features = []
for files in image_id:
    im, mask = load_image_and_mask(files, data_path=data_path)
    features.append(asymmetry(mask))
    print(im)

df_features['FEATURE_A'] = features
# HERE COLORS BUT I STILL KINDA DONT KNOW


#SAVING THEM TO THE FILE
df_features.to_csv(csv_path, index=False)

print('SUCCES!')