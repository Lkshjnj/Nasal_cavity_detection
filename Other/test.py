import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm.keras import TqdmCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam

base_path = '../input/face-images-with-marked-landmark-points/'

df = pd.read_csv(base_path + 'facial_keypoints.csv')
df.head()

df = df.fillna(0)
# split columns two by two
c = 0
columns = {}
temp = []
for i, e in enumerate(list(df.columns)):
    temp.append(e)
    c += 1
    if c == 2:
        columns[e.split('_y')[0]] = temp
        temp = []
        c = 0
columns.keys()

keypoints_dict = {}
for k in columns.keys():
    keypoints_dict[k] = df[columns[k]].values
keypoints = np.array(list(keypoints_dict.values()))
keypoints = np.swapaxes(keypoints, 0, 1)

features = np.load(base_path + 'face_images.npz')['face_images']
features = np.swapaxes(np.swapaxes(features, 1, 2), 0, 1)

def display_image_keypoints(nr, features, keypoints):
    plt.imshow(features[nr])
    for i in range(keypoints.shape[1]):
        element = keypoints[nr,i,:]
        plt.scatter(element[0],element[1],c='r',s=12)
display_image_keypoints(3, features, keypoints)