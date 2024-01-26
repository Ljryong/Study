# 이미지 데이터 150 150 3 고정
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


#1
train_datagen = ImageDataGenerator(1./255)

path_train = 'c:/_data/image/rps'

xy_train = train_datagen.flow_from_directory(path_train,
                                             class_mode='categorical',
                                             color_mode='rgb',
                                             batch_size=30,
                                             target_size=(150,150))

x = []
y = []
for i in range(len(xy_train)) : 
    a, b = xy_train.next()
    x.append(a)
    y.append(b)

x = np.concatenate(x,axis=0)
y = np.concatenate(y,axis=0)


print('train data ok')

np_path = 'c:/_data/_save_npy//'

np.save(np_path + 'keras39_09_x.npy' , arr= x)
np.save(np_path + 'keras39_09_y.npy' , arr= y)

