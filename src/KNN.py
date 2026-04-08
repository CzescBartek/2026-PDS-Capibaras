import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load features
df = pd.read_csv('../data/features.csv')

# Split into features and labels
feature_cols = [
    'FEATURE_A',
    'FEATURE_B_R', 'FEATURE_B_G', 'FEATURE_B_B',
    'FEATURE_BORDER_COMPACTNESS', 'FEATURE_BORDER_CONVEXITY'
]
