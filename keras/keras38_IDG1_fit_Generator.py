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


#1
train_datagen = ImageDataGenerator(rescale=1./255, )
                                #    horizontal_flip= True , vertical_flip = True ,
                                #    width_shift_range = 0.1,height_shift_range = 0.1,
                                #    rotation_range = 5 , zoom_range = 1.2,
                                #    shear_range = 0.7 , 
                                #    fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

path_train = 'c:/_data/image/brain/train//'
path_test = 'c:/_data/image/brain/test//'


xy_train = train_datagen.flow_from_directory(path_train, 
                                             target_size = (100,100),
                                             batch_size = 10,        # 통배치 하는 이유는 데이터가 작아서 커지면 커질수록 통배치는 사용 X 
                                                                     # xy_test의 batch_size 는 fit 에 들어가는게 아니라서 밑에와 달라도 상관 없다
                                                                     # 밑에 batch 는 이미지를 자르는 것뿐이다.
                                             class_mode='binary', 
                                             color_mode= 'grayscale',
                                             shuffle=True)


xy_test = test_datagen.flow_from_directory(                          
                                             path_test,              
                                             target_size = (100,100) ,
                                             batch_size =  10, 
                                             class_mode='binary',
                                             shuffle=False,          
                                             color_mode= 'grayscale'
                                             )


# print(xy_train[0][0])      
# print(xy_train[0][1])      
print(xy_train[0][0][0].shape)     # (160, 100, 100, 1)
print(xy_train[0][1][0].shape)     # (160,)



# x_train = xy_train[0][0]  
# y_train = xy_train[0][1]






#배치로 잘린 데이터 합치기    / 선의형
# x_train = []
# y_train = []
# for i in range(xy_train.samples // xy_train.batch_size):
#     batch = next(xy_train)
#     x_train.append(batch[0])
#     y_train.append(batch[1])
# x_train = np.concatenate(x_train)
# y_train = np.concatenate(y_train)
'''

# print(xy_train.next())
es = EarlyStopping(monitor='val_loss' , mode = 'min' , patience= 15 , restore_best_weights=True , verbose= 1  )


#2 모델구성
model = Sequential()
model.add(Conv2D(94,(3,3),input_shape = (100,100,1), padding='valid' , strides=1 , activation='relu' ))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(12,(2,2), activation='relu' ))
model.add(MaxPooling2D())

model.add(Conv2D(81,(2,2), activation='relu' ))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(9,(3,3), activation='relu' ))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(73,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(12))

model.add(Dense(42,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))

# model.summary()


#3 컴파일, 훈련
model.compile(loss= 'binary_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit_generator(xy_train, epochs = 100  , verbose= 1 ,callbacks=[es], validation_data = xy_test , steps_per_epoch=17 )
                    # batch_size= 10 // batch 사이즈를 위에서 지정해주고 fit에서 지정하지 않는다.
                    # validation_data = xy_test // validation 스플릿 대신에 쓰는것
                    # UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
                    # steps_per_epoch = 전체 데이터/batch = 160/10 = 16 // 이보다 높은 숫자는 안됨 / 낮은 숫자는 잘라서 버린다.
                    # 17은 error // 15는 데이터 손실
                    
                    
                    
# model.fit(xy_train, epochs = 100  , verbose= 1 ,callbacks=[es], validation_data = xy_test  )
# validation 스플릿은 텐서와 넘파이만 받을 수 있어서 인터레이터 형태를 받지 못한다.
# == fit 안에 fit_generator 가 들어가있다. generator 를 쓰지 않아도 fit으로 쓸 수 있다.
# batch_size= 10 // fit 안에 썻어도 위에 쓴 batch로 들어간다 / fit에서는 써도 먹히지 않는다.      

              


#4 평가, 예측
loss = model.evaluate_generator(xy_test)
y_predict = model.predict_generator(xy_test)

print('loss',loss)

y_predict = np.round(y_predict)




# time :  128.8634068965912  (150,150)

# time :  82.54952311515808  (100,100)

# Epoch 10/10       (10,10)
# 1120/1120 [==============================] - 2s 2ms/step - loss: 0.6480 - acc: 0.6103 - val_loss: 0.6590 - val_acc: 0.5979
# 188/188 [==============================] - 0s 2ms/step - loss: 0.6548 - acc: 0.6062
# 188/188 [==============================] - 0s 621us/step
# loss [0.6548140645027161, 0.606166660785675]
# Acc =  0.6061666666666666
# time :  44.61295771598816



# Epoch 10/10       (80,80)
# 1120/1120 [==============================] - 7s 6ms/step - loss: 0.5977 - acc: 0.6718 - val_loss: 0.6838 - val_acc: 0.5879
# 188/188 [==============================] - 1s 4ms/step - loss: 0.6665 - acc: 0.6025
# 188/188 [==============================] - 1s 3ms/step
# loss [0.6664801239967346, 0.6025000214576721]
# Acc =  0.6025
# time :  69.61687755584717




# Epoch 83: early stopping
# 188/188 [==============================] - 1s 5ms/step - loss: 0.6887 - acc: 0.5430
# 188/188 [==============================] - 1s 5ms/step
# loss [0.6886634230613708, 0.5429999828338623]
# Acc =  0.543
# time :  75.94417262077332

# Epoch 38: early stopping
# 188/188 [==============================] - 1s 5ms/step - loss: 0.6522 - acc: 0.6113
# 188/188 [==============================] - 1s 4ms/step
# loss [0.6521573662757874, 0.6113333106040955]
# Acc =  0.6113333333333333
# time :  76.63124442100525


# loss [0.6932511925697327, 0.5005263090133667]
# Acc =  0.5005263157894737
# time :  23.81118869781494







'''