import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor, pi
from FEATURE_A import asymmetry, midpointGroup9
import os
from FEATURE_COLOR import slic_segmentation, get_rgb_means, load_image_and_mask
from FEATURE_BORDER import compactness_score, convexity_score
from FEATURE_LBP import result_LBP
from hair_coverage import hair_coverage
from hair_remove import removeHair_auto
from penmark_remove import remove_penmarks
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
import cv2

df = pd.read_csv("../data/metadata.csv")
image_id = df["img_id"].tolist()

data_path = '../data/'
imgs_path = "../data/imgs/"
mask_path= "../data/masks/"
csv_path = '../data/features.csv'

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
hair = []
features = []
Lbp_list = []
i = 0
for files in image_id:
    value = df.loc[df['img_id'] == files, 'diagnostic'].values

    if len(value) > 0 and value[0] in ['BCC', 'MEL', 'SCC']:
        cancerous.append(1)
    else:
        cancerous.append(0)
    #This for color
    im, mask = load_image_and_mask(files, data_path=data_path)

    path = '../data/imgs/' + files
    img_org = cv2.imread(path)
    img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

    img1 = remove_penmarks(img_org)
    img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img_out = removeHair_auto(img1, img_gray1)
    im = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    img_gray2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # if im.shape[:2] != mask.shape[:2]:
    #     mask_uint8 = mask.astype(np.uint8) * 255
    #     mask = cv2.resize(mask_uint8, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)

    # means = get_rgb_means(im, mask)
    # if means is not None and len(means) > 0:
    #     colors = np.mean(means, axis=0)
    # else:
    #     colors = np.array([0, 0, 0])

    # features.append(asymmetry(mask))
    # compactness.append(compactness_score(mask))
    # convexity.append(convexity_score(mask))
    # r.append(colors[0])
    # g.append(colors[1])
    # b.append(colors[2])
    hair.append(hair_coverage(img_gray2))
    Lbp = result_LBP(im)
    Lbp_list.append(Lbp)

    print(f'{i}. ', end='')
    print(files)
    i+=1
    

# df_features['FEATURE_A'] = features
# df_features['FEATURE_B_R'] = r
# df_features['FEATURE_B_G'] = g
# df_features['FEATURE_B_B'] = b
# df_features['FEATURE_BORDER_COMPACTNESS'] = compactness
# df_features['FEATURE_BORDER_CONVEXITY'] = convexity
# df_features['Cancerous'] = cancerous


df_features['Hair'] = hair


df_features['LBP'] = Lbp_list
df_features.to_csv(csv_path, index=False)

print('SUCCES!')