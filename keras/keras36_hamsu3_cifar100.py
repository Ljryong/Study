from keras.datasets import cifar100
from keras.models import Sequential , Model
from keras.layers import Dense , Conv2D , Dropout , Flatten , MaxPooling2D , BatchNormalization , Input
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#1

(x_train,y_train), (x_test,y_test) = cifar100.load_data()

# print(x_train.shape,y_train.shape)      # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape,y_test.shape)        # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000,3072)
x_test = x_test.reshape(10000,3072)
# print(np.unique(y_test))                # 100

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train)
# print(y_test)

es = EarlyStopping(monitor='val_loss' , mode = 'min' , patience = 100 ,  restore_best_weights= True , verbose= 1  )

#2 모델구성
# model = Sequential()
# model.add(Dense(2500 , input_shape = (3072,) , activation='relu' ))
# model.add(Dense(2600,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(2100,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(2200,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1700,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1800,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1300,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(800,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(900,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(300,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(200,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(100,activation='softmax'))


#2-1
input = Input(shape=(3072,))
d1 = Dense(2500,activation='relu')(input)
drop1 = Dropout(0.2)(d1)
d2 = Dense(2600,activation='relu')(drop1)
drop2 = Dropout(0.2)(d2)
d3 = Dense(2100,activation='relu')(drop2)
drop3 = Dropout(0.2)(d3)
d4 = Dense(2200,activation='relu')(drop3)
drop4 = Dropout(0.2)(d4)
d5 = Dense(1700,activation='relu')(drop4)
drop5 = Dropout(0.2)(d5)
d6 = Dense(1800,activation='relu')(drop5)
drop6 = Dropout(0.2)(d6)
d7 = Dense(1300,activation='relu')(drop6)
drop7 = Dropout(0.2)(d7)
d8 = Dense(800,activation='relu')(drop7)
drop8 = Dropout(0.2)(d8)
d9 = Dense(900,activation='relu')(drop8)
drop9 = Dropout(0.2)(d9)
d10 = Dense(300,activation='relu')(drop9)
drop10 = Dropout(0.2)(d10)
d11 = Dense(200,activation='relu')(drop10)
drop11 = Dropout(0.2)(d11)
output = Dense(100,activation='softmax')(drop11)
model = Model(inputs = input , outputs = output)







#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train,epochs = 10 , batch_size= 1000 , validation_split=0.2 , callbacks= [es]  , verbose= 1)

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print('loss = ' , loss)

y_test = np.argmax(y_test,axis=1)
y_predict = np.argmax(y_predict,axis=1)

print('acc',accuracy_score(y_test,y_predict))


# Epoch 113: early stopping
# 313/313 [==============================] - 0s 1ms/step - loss: 4.5487 - acc: 0.0195 
# 313/313 [==============================] - 0s 830us/step
# loss =  [4.54869270324707, 0.019500000402331352]
# acc 0.0195



# 함수
# Epoch 10/10
# 40/40 [==============================] - 2s 40ms/step - loss: 4.5572 - acc: 0.0164 - val_loss: 4.5157 - val_acc: 0.0182
# 313/313 [==============================] - 1s 2ms/step - loss: 4.5110 - acc: 0.0184
# 313/313 [==============================] - 0s 1ms/step
# loss =  [4.510982513427734, 0.018400000408291817]
# acc 0.0184