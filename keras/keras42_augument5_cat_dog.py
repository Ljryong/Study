import numpy as np
from keras.preprocessing.image import ImageDataGenerator

#1 데이터

train_datagen = ImageDataGenerator(1./255)

test_datagen = ImageDataGenerator(1./255)

path_train = 'C:\_data\image\catdog\Train\\'
path_test = 'C:\_data\image\catdog\Test\\'


xy_train = train_datagen.flow_from_directory(
    path_train,
    
)










