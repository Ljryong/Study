# x,y 추출해서 모델 만들기
# 성능 0.99 이상

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense , Conv2D ,Flatten , MaxPooling2D , Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


#1 데이터

train_datagen = ImageDataGenerator(rescale=1/255.,
                                #    horizontal_flip=True,     # 수평 뒤집기
                                #    vertical_flip=True,       # 수직 뒤집기
                                #    width_shift_range=0.1,    # 평행이동
                                #    height_shift_range=0.1,   # 평행이동
                                #    rotation_range=5,         # 정해진 각도만큼 이미지를 회전
                                #    zoom_range=1.2,           # 축소 또는 확대
                                #    shear_range=0.7,          # 좌표하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환
                                   fill_mode='nearest'       # 너의 빈자리를 가장 비슷한 책으로 채워
                                                             # 0로 나타내 주는것도 잇으니 찾아보기
                                    
                                   )

test_datagen = ImageDataGenerator(rescale=1./255)      # test 데이터는 train에서 훈련한 것과 비교하는 실제 데이터로 해야되기 때문에 rescale만 쓴다.

path_train = 'c:/study/image/brain/train//'
path_test = 'c:/study/image/brain/test//'


xy_train = train_datagen.flow_from_directory(                                   # 그림들을 가져와서 수치화해주는 것 (이터레이터형태)
                                             path_train,
                                             target_size = (150,150) ,          # 내가 정한 수치까지 그림들의 사이즈를 줄이거나 늘린다// 
                                             batch_size =  160,                  # 160 이상의 수를 쓰면 x의 통 데이터(160)로 들어간다
                                             class_mode='binary',
                                             shuffle=True 
                                             )       


# Found 160 images belonging to 2 classes.


# from sklearn.datasets import load_diabetes
# datasets = load_diabetes()
# print(datasets)

xy_test = test_datagen.flow_from_directory(                            
                                             path_test,
                                             target_size = (150,150) ,   
                                             batch_size =  160, 
                                             class_mode='binary',
                                             shuffle=False              # test는 섞는것이 아니다 // 건드리면 데이터 조작이다.
                                             )       

# Found 120 images belonging to 2 classes.

# print(xy_train)
 # <keras.preprocessing.image.DirectoryIterator object at 0x000001CC26C83520>

# print(xy_train.next())              # 1번째 값만 보여준다
# print(xy_train[0])
# print(xy_train[16])               # 에러 : 전체데이터/batch_size = 10 이라서 160/10은 16개인데 
                                    #       [16]는 17번째의 값을 빼라고 해서 에러가 나는것이다.

# print(xy_train[0][0])       # 첫번째의 배치 x
# print(xy_train[0][1])       # 두번째의 배치 y
# print(xy_train[0][0].shape)             # (10, 200, 200, 3) 흑백은 칼라다 o //칼라는 흑백이다 X
# print(xy_train[0][1].shape)                 # (160,)

# print(type(xy_train))           # 이터레이터 형태
# print(type(xy_train[0]))        # <class 'tuple'>
# print(type(xy_train[0][0]))     # <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))     # <class 'numpy.ndarray'>


x_train = xy_train[0][0]            # 데이터를 이렇게 넣으려면 통배치로 바꿔서 넣어야 된다.
x_test = xy_test[0][0]
y_train = xy_train[0][1]
y_test = xy_test[0][1]

# 2진 분류는 onehot을 쓰지 않는다.

# x_train = x_train.reshape(120000,)

es = EarlyStopping(monitor='val_loss' , mode = 'min' , patience= 70 , restore_best_weights=True , verbose= 1  )


#2 모델구성
model = Sequential()
model.add(Conv2D(108,(2,2),input_shape = (150,150,3), padding='valid' , strides=1 , activation='relu' ))
model.add(MaxPooling2D())
model.add(Conv2D(12,(3,3), activation='relu' ))
model.add(Conv2D(97,(2,2), activation='relu' ))
model.add(Dropout(0.2))
model.add(Conv2D(13,(3,3) ))
model.add(Flatten())
model.add(Dense(91,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8))
model.add(Dense(30,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

# model.summary()

#3 컴파일, 훈련
model.compile(loss= 'binary_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train, epochs = 10000 , batch_size= 500 , validation_split= 0.2, verbose= 1 ,callbacks=[es])


#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print('loss',loss)

y_test = np.round(y_test)
y_predict = np.round(y_predict)

print('Acc = ' , accuracy_score(y_test,y_predict))




# Epoch 129: early stopping
# 4/4 [==============================] - 0s 72ms/step - loss: 0.3958 - acc: 0.8250
# 4/4 [==============================] - 0s 11ms/step
# loss [0.39580318331718445, 0.824999988079071]
# Acc =  0.825


# Epoch 178: early stopping
# 4/4 [==============================] - 0s 83ms/step - loss: 0.0509 - acc: 0.9750
# 4/4 [==============================] - 0s 13ms/step
# loss [0.05093661695718765, 0.9750000238418579]
# Acc =  0.975

# Epoch 192: early stopping
# 4/4 [==============================] - 0s 89ms/step - loss: 0.0257 - acc: 0.9917
# 4/4 [==============================] - 0s 11ms/step
# loss [0.025711089372634888, 0.9916666746139526]
# Acc =  0.9916666666666667