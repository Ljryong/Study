from keras.datasets import cifar10
from keras.models import Sequential 
from keras.layers import Dense , Conv2D , Dropout , Flatten , MaxPooling2D 
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical

#1 데이터
(x_train,y_train) , (x_test,y_test) = cifar10.load_data()

print(x_train.shape,y_train.shape)      # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape)        # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_test))
x_train = x_train.reshape(50000,3072)
x_test = x_test.reshape(10000,3072)      # (1600000, 96) (320000, 96)

print(x_train.shape,x_test.shape)       # (50000, 3072) (10000, 3072)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train,y_test) 

es = EarlyStopping(monitor='val_loss' , mode='min' , patience= 100 , verbose=1 , restore_best_weights= True  )


#2 모델구성
model = Sequential()
model.add(Dense(2500 , input_shape = (3072,) , activation='relu' ))
model.add(Dense(2600,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2200,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1700,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1800,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1300,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(800,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(900,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(300,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(400,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

#3 컴파일, 훈련
model.compile(loss='categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train , epochs=10000 , batch_size=1000 , validation_split=0.2 , callbacks=[es], verbose=1 )


#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print('loss',loss)
y_test = np.argmax(y_test,axis=1)
y_predict = np.argmax(y_predict,axis=1)

print(accuracy_score(y_test,y_predict))




# Epoch 170: early stopping
# 313/313 [==============================] - 1s 1ms/step - loss: 1.5239 - acc: 0.4750
# 313/313 [==============================] - 0s 908us/step
# loss [1.523861289024353, 0.4749999940395355]
# 0.475


