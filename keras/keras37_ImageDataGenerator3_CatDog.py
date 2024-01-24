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
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   horizontal_flip= True , vertical_flip = True ,
                                   width_shift_range = 0.1,height_shift_range = 0.1,
                                   rotation_range = 5 , zoom_range = 1.2,
                                   shear_range = 0.7 , fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

path_train = 'c:/_data/image/catdog/Train//'
path_test = 'c:/_data/image/catdog/Test//'


xy_train = train_datagen.flow_from_directory(path_train, 
                                             target_size = (150,150),
                                             batch_size = 20000,
                                             class_mode='binary', 
                                             shuffle=True)

x_train = xy_train[0][0]  
y_train = xy_train[0][1]

x_train , x_test, y_train , y_test = train_test_split(x_train,y_train , test_size=0.3 , random_state= 0 ,  )




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

end = time.time()

print('time : ' , end - start )



