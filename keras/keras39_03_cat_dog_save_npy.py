# 과제

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense , Conv2D ,Flatten , MaxPooling2D , Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import time

############################ 사진이 꺠져있음 , 확인방법 #########
# 1. 사진의 파일 크기를 확인하여 제거할 수 있다.




start = time.time()
#1
train_datagen = ImageDataGenerator(rescale=1./255, )
                                #    horizontal_flip= True , vertical_flip = True ,
                                #    width_shift_range = 0.1,height_shift_range = 0.1,
                                #    rotation_range = 5 , zoom_range = 1.2,
                                #    shear_range = 0.7 , 
                                #    fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

path_train = 'c:/_data/image/catdog/Train//'
path_test = 'c:/_data/image/catdog/Test//'


xy_train = train_datagen.flow_from_directory(path_train, 
                                             target_size = (100,100),
                                             batch_size = 1000,        # 통배치 하는 이유는 데이터가 작아서 커지면 커질수록 통배치는 사용 X 
                                             class_mode = 'binary', 
                                          #  color_mode= 'grayscale',
                                             shuffle=True)

test = test_datagen.flow_from_directory(path_test, 
                                             target_size = (100,100),
                                             batch_size = 1000,      
                                             class_mode =None, 
                                          #  color_mode = 'grayscale',
                                             shuffle=False)




#배치로 잘린 데이터 합치기    / 선의형

#배치로 잘린 데이터 합치기 // 정훈이 형 
x = []
y = []
for i in range(len(xy_train)):
    batch, count  = xy_train.next()
    x.append(batch)
    y.append(count)
    
    # all_images와 all_labels을 numpy 배열로 변환하면 하나의 데이터로 만들어진 것입니다.
x = np.concatenate(x, axis=0)
y = np.concatenate(y, axis=0)

print(x.shape)
print(y.shape)

submit = []
for i in range(len(test)):
    images = test.next()
    submit.append(images)

submit = np.concatenate(submit,axis=0)

# print(xy_test[0][0].shape)
print(submit.shape)



np_path = 'c:/_data/_save_npy//'
np.save(np_path + 'keras39_catdog_x_train.npy', arr = x)
np.save(np_path + 'keras39_catdog_y_train.npy', arr = y)
np.save(np_path + 'keras39_catdog_test.npy', arr = submit )

