from keras.models import Sequential, Model
from keras.layers import Dense , Input , Dropout, MaxPooling2D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

# 이미지 데이터 300 300 3 고정

# 1 
np_path = 'c:/_data/_save_npy//'

train_datagan = ImageDataGenerator(1./255)

path_train = 'c:/_data/image/horse_human/'

xy_train = train_datagan.flow_from_directory(path_train,
                                             shuffle=True , 
                                             class_mode='binary',
                                             color_mode= 'rgb',
                                             target_size=(300,300),                                             
                                             batch_size= 30)

x= []
y= []
for i in range(len(xy_train)) : 
    a , b = xy_train.next()
    x.append(a)
    y.append(b)

x = np.concatenate(x, axis= 0)
y = np.concatenate(y, axis= 0)

np.save(np_path + 'keras39_11_x.npy' , arr = x )
np.save(np_path + 'keras39_11_y.npy' , arr = y )
