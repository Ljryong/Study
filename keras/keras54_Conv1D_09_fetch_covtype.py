from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential , Model
from keras.layers import Dense , Dropout , Input , Conv2D , MaxPooling2D , Flatten , LSTM , Conv1D
from keras.callbacks import EarlyStopping , ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
import time

#1
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape,y.shape)      # (581012, 54) (581012,)
print(pd.value_counts(y))   # 2    283301 , 1    211840 , 3     35754 , 7     20510 , 6     17367 , 5      9493 , 4      2747   (n,7)

from keras.utils import to_categorical          # 라벨링을 할 때 0이 포함된다.
one_hot = to_categorical(y-1)
print(one_hot)

# one_hot = pd.get_dummies(y)

# from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(-1,1)
# ohe = OneHotEncoder()
# ohe.fit(y)
# one_hot = ohe.transform(y).toarray()

# print(one_hot)
x = x.reshape(-1,9,6)


x_train , x_test , y_train , y_test = train_test_split(x,one_hot,test_size=0.3 , random_state= 2 ,shuffle=True, stratify=y ) # 0

es= EarlyStopping(monitor='val_loss' , mode = 'min', verbose= 1 ,patience=10, restore_best_weights=True )

date = datetime.datetime.now()
date = date.strftime('%m%d-%H%M')
path = 'c:/_data/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path , 'k28_9_', date , '_', filename ])

# print(datasets.DESCR)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

mcp = ModelCheckpoint(monitor='val_loss', mode='min' , verbose=1, save_best_only=True , filepath= filepath   )


#2
# model = Sequential()
# model.add(Dense(1024,input_shape = (54,)))
# model.add(Dense(512))
# model.add(Dropout(0.2))
# model.add(Dense(256))
# model.add(Dropout(0.7))
# model.add(Dense(128))
# model.add(Dropout(0.3))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(7,activation='softmax'))


#2-1
# input = Input(shape=(54,))
# d1 = Dense(1024)(input)
# d2 = Dense(512)(d1)
# drop1 = Dropout(0.2)(d2)
# d3 = Dense(256)(drop1)
# drop2 = Dropout(0.7)(d3)
# d4 = Dense(128)(drop2)
# drop3 = Dropout(0.3)(d4)
# d5 = Dense(64)(drop3)
# d6 = Dense(32)(d5)
# output = Dense(7,activation='softmax')(d6)

# model = Model(inputs = input , outputs = output) 


#2-2
model = Sequential()
model.add(Conv1D(20,2,input_shape = (9,6)))
model.add(Conv1D(10,2))
model.add(Flatten())
model.add(Dense(12))
model.add(Dense(36))
model.add(Dense(7,activation='softmax'))



#3
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])  # 오류 뜨는거 해결을 위해 sparse_categorical_crossentropy를 넣어봄
start_time = time.time()

model.fit(x_train, y_train, epochs = 100 , batch_size=100000, verbose=1 , validation_split=0.2, callbacks=[es,mcp] )
end_time = time.time()



#4
result = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test , axis =1)
y_predict = np.argmax(y_predict, axis = 1)

print(y_test.shape)
print(y_predict.shape)

acc = accuracy_score(y_test,y_predict)

print('acc = ' , acc)
print('reslut = ',result)
print('시간 : ',end_time - start_time)


# Epoch 35: early stopping
# 5447/5447 [==============================] - 7s 1ms/step - loss: 0.6882 - acc: 0.7010
# 5447/5447 [==============================] - 7s 1ms/step
# (174304,)
# (174304,)
# acc =  0.7009707178263265
# batch = 2500

# 0
# Epoch 22: early stopping
# 5447/5447 [==============================] - 7s 1ms/step - loss: 0.6842 - acc: 0.7016
# 5447/5447 [==============================] - 6s 1ms/step
# (174304,)
# (174304,)
# acc =  0.7015673765375436
# batch = 1000

# 2
# Epoch 22: early stopping
# 5447/5447 [==============================] - 6s 1ms/step - loss: 0.6818 - acc: 0.7041
# 5447/5447 [==============================] - 6s 1ms/step
# (174304,)
# (174304,)
# acc =  0.7041146502662016
# batch = 1000


# MinMaxScaler
# Epoch 93: early stopping
# 5447/5447 [==============================] - 5s 831us/step - loss: 0.6298 - acc: 0.7239
# 5447/5447 [==============================] - 4s 785us/step
# (174304,)
# (174304,)
# acc =  0.7239191297962181
# reslut =  [0.6298187375068665, 0.723919153213501]

# StandardScaler
# Epoch 58: early stopping
# 5447/5447 [==============================] - 4s 806us/step - loss: 0.6309 - acc: 0.7228
# 5447/5447 [==============================] - 4s 806us/step
# (174304,)
# (174304,)
# acc =  0.7227774463007159
# reslut =  [0.6309492588043213, 0.7227774262428284]


# MaxAbsScaler
# Epoch 108: early stopping
# 5447/5447 [==============================] - 5s 842us/step - loss: 0.6302 - acc: 0.7242
# 5447/5447 [==============================] - 4s 804us/step
# (174304,)
# (174304,)
# acc =  0.7242002478428493
# reslut =  [0.630240261554718, 0.7242002487182617]

# RobustScaler
# Epoch 51: early stopping
# 5447/5447 [==============================] - 9s 2ms/step - loss: 0.6287 - acc: 0.7247
# 5447/5447 [==============================] - 8s 1ms/step
# (174304,)
# (174304,)
# acc =  0.7246764273912245
# reslut =  [0.6287251114845276, 0.7246764302253723]


# cpu
# 시간 :  2574.008665084839
# gpu
# 시간 :  396.80502557754517


# Cnn
# acc =  0.6863583165044979
# reslut =  [0.71393883228302, 0.6863583326339722]
# 시간 :  16.293997764587402



# LSTM
# Epoch 100: val_loss improved from 0.79014 to 0.78924, saving model to c:/_data/_save/MCP\k28_9_0130-1146_0100-0.7892.hdf5
# 4/4 [==============================] - 0s 73ms/step - loss: 0.7935 - acc: 0.6367 - val_loss: 0.7892 - val_acc: 0.6374
# 5447/5447 [==============================] - 8s 1ms/step - loss: 0.7894 - acc: 0.6367
# 5447/5447 [==============================] - 5s 894us/step
# (174304,)
# (174304,)
# acc =  0.6367209014136221
# reslut =  [0.7894095182418823, 0.6367208957672119]
# 시간 :  29.61864185333252

# Conv1D
# Epoch 80: early stopping
# 5447/5447 [==============================] - 6s 1ms/step - loss: 1.4231 - acc: 0.5509
# 5447/5447 [==============================] - 3s 625us/step
# (174304,)
# (174304,)
# acc =  0.5508938406462273
# reslut =  [1.4231102466583252, 0.5508938431739807]
# 시간 :  15.65010380744934







