import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor, pi
from FEATURE_A import asymmetry, midpointGroup9
import os

df = pd.read_csv("../data/metadata.csv")
image_id = df["img_id"].tolist()

def save_feature(feature):
    imgs_path = "../data/imgs/"
    mask_path= "../data/masks/"
    features = []
    for files in image_id:
        file_im = imgs_path + files
        file_mask = (mask_path + files).replace(".png", "_mask.png")
        im = plt.imread(file_im)
        mask = plt.imread(file_mask)
        features.append(feature(mask))
    return features


csv_path = '../data/features.csv'

# CHECKING IF THE FILE EXISTS OR IS IT EMPTY
if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:

    df_features = pd.read_csv(csv_path)
else:

    df_features = pd.DataFrame({'img_id': image_id})


# LOADING FEATURE A INTO FEATURES.CSV
df_features['FEATURE_A'] = save_feature(asymmetry)


#SAVING THEM TO THE FILE
df_features.to_csv(csv_path, index=False)

print('SUCCES!')