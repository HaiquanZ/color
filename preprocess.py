import cv2
import os
import glob
import pickle
from tqdm import tqdm
import numpy as np
import sys

def split_hsv_img(img_path):
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (224, 224))
    img_gray = cv2.imread(img_path, 0)
    img_gray = cv2.resize(img_gray, (224, 224))
    img_gray = np.repeat(img_gray[:, :, np.newaxis], 3, axis=2)
    return img_gray, img

fns = os.listdir('data_raw')
X, y = [], []
for count, fn in enumerate(fns):
    if count>-1:
        hs, v = split_hsv_img(os.path.join('data_raw', fn))
        X.append(hs)
        y.append(v)


with open('data_processed/X_0.pkl', 'wb') as f:
    pickle.dump(X, f)
with open('data_processed/y_0.pkl', 'wb') as f:
    pickle.dump(y, f)
print(np.shape(X), np.shape(y))
