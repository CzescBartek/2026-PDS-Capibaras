import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor, pi
from FEATURE_A import asymmetry, midpointGroup9
import os
from FEATURE_COLOR import slic_segmentation, get_rgb_means, load_image_and_mask
from FEATURE_BORDER import compactness_score, convexity_score
import cv2
import numpy as np
from math import sqrt, floor, ceil, nan, pi
from skimage import color, exposure
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from statistics import variance, stdev
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

r = []
g = []
b = []

compactness = []
convexity = []
cancerous = []

features = []
for files in image_id:
    value = df.loc[df['img_id'] == files, 'diagnostic'].values
    
    if len(value) > 0 and value[0] in ['BCC', 'MEL', 'SCC']:
        cancerous.append(1)
    else:
        cancerous.append(0)

    im, mask = load_image_and_mask(files, data_path=data_path)
    features.append(asymmetry(mask))
    means = get_rgb_means(im, mask)
    if means is not None and len(means) > 0:
        colors = np.mean(means, axis=0)
    else:
        colors = np.array([0, 0, 0])

    #Border Features Here 
    compactness.append(compactness_score(mask))
    convexity.append(convexity_score(mask))
    r.append(colors[0])
    g.append(colors[1])
    b.append(colors[2])
    print(im)


df_features['FEATURE_A'] = features
df_features['FEATURE_B_R'] = r
df_features['FEATURE_B_G'] = g
df_features['FEATURE_B_B'] = b
#Here border too
df_features['FEATURE_BORDER_COMPACTNESS'] = compactness
df_features['FEATURE_BORDER_CONVEXITY'] = convexity
#Labels
df_features['Cancerous'] = cancerous

# HERE COLORS BUT I STILL KINDA DONT KNOW


#SAVING THEM TO THE FILE
df_features.to_csv(csv_path, index=False)

print('SUCCES!')