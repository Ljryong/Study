from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential , Model 
from keras.layers import Dense , Dropout , Input , Conv2D , Flatten , LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time

#1 데이터

datasets = fetch_california_housing()
print(datasets)

x = datasets.data
y = datasets.target

print(x.shape)      # (20640, 8)
print(y.shape)      # (20640,)

x = x.reshape(20640,4,2)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 0 , shuffle= True  )

from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler, StandardScaler, RobustScaler
# slr = MaxAbsScaler()
# slr = MinMaxScaler()
# slr = StandardScaler()
slr = RobustScaler()

# slr.fit(x_train)
# x_train = slr.transform(x_train)
# x_test = slr.transform(x_test)

date = datetime.datetime.now()
path = 'c:/_data/_save/MCP/'
date = date.strftime('%m%d-%H%M')
filename = '{epoch:04d}-{val_loss:04f}.hdf5'
filepath = ''.join([path,'k28_2_',date,"_",filename])


#2 모델구성
# model = Sequential()
# model.add(Dense(5,input_shpae = (8,)))
# model.add(Dropout(0.2))
# model.add(Dense(1))

#2-1
# input = Input(shape =(8,) )
# d1 = Dense(5)(input)
# drop1 = Dropout(0.2)(d1)
# output = Dense(1)(drop1)
# model = Model(inputs = input , outputs =output )
# model.summary()

#2-2 
model = Sequential()
model.add(LSTM(50,input_shape = (4,2)))
model.add(Dense(20))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer= 'adam', metrics=['mse','mae'] )

from keras.callbacks import EarlyStopping , ModelCheckpoint
es = EarlyStopping(monitor='val_loss' , mode = 'min' , patience= 10 , verbose = 1 , restore_best_weights=True )
mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'min' , verbose= 1 ,save_best_only=True , filepath = filepath )
start_time = time.time()
hist = model.fit(x_train,y_train, epochs= 1000 ,batch_size= 100, validation_split=0.2 , callbacks = [es,mcp])
end_time = time.time()


#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)



# plt.figure(figsize = (9,6))
# plt.plot(hist.history['loss'], c = 'red' , label = 'loss' , marker = '.')
# plt.plot(hist.history['val_loss'],c = 'blue' , label = 'val_loss' , marker = '.')
# plt.legend(loc = 'upper right')


# print(hist)
# plt.title('california loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()
# plt.show()


print(loss)
print(r2)
print('시간 :' , end_time - start_time)


# Epoch 220: early stopping
# 194/194 [==============================] - 0s 368us/step - loss: 0.6333
# 194/194 [==============================] - 0s 349us/step
# 0.6333415508270264 = loss
# 0.5249593856949173 = r2

# MaxAbsScaler
# Epoch 1122: early stopping
# 194/194 [==============================] - 0s 421us/step - loss: 0.6213 - mse: 0.6213 - mae: 0.5649
# 194/194 [==============================] - 0s 323us/step
# [0.621303379535675, 0.621303379535675, 0.5649369359016418]
# 0.5339887470787599

# MinMaxScaler
# Epoch 233: early stopping
# 194/194 [==============================] - 0s 425us/step - loss: 0.5461 - mse: 0.5461 - mae: 0.5390
# 194/194 [==============================] - 0s 392us/step
# [0.5461402535438538, 0.5461402535438538, 0.5389795899391174]
# 0.5903652399404002

# StandardScaler
# Epoch 57: early stopping
# 194/194 [==============================] - 0s 427us/step - loss: 0.5477 - mse: 0.5477 - mae: 0.5384
# 194/194 [==============================] - 0s 382us/step
# [0.547719419002533, 0.547719419002533, 0.5384447574615479]
# 0.589180695795096

# RobustScaler
# Epoch 1122: early stopping
# 194/194 [==============================] - 0s 421us/step - loss: 0.6213 - mse: 0.6213 - mae: 0.5649
# 194/194 [==============================] - 0s 323us/step
# [0.621303379535675, 0.621303379535675, 0.5649369359016418]
# 0.5339887470787599


# Dropout
# Epoch 99: val_loss did not improve from 0.50111
# 116/116 [==============================] - 0s 750us/step - loss: 0.5944 - mse: 0.5944 - mae: 0.5652 - val_loss: 0.5046 - val_mse: 0.5046 - val_mae: 0.5317
# Epoch 99: early stopping
# 194/194 [==============================] - 0s 435us/step - loss: 0.5698 - mse: 0.5698 - mae: 0.5393
# 194/194 [==============================] - 0s 377us/step
# [0.5697676539421082, 0.5697676539421082, 0.5393381714820862]
# 0.5726433460952995

# cpu
# 시간 : 85.8211178779602
# gpu
# 시간 : 263.6768412590027

# Cnn
# Epoch 48: early stopping
# 194/194 [==============================] - 0s 1ms/step - loss: 1.1668 - mse: 1.1668 - mae: 0.8210
# 194/194 [==============================] - 0s 569us/step
# [1.166846513748169, 1.166846513748169, 0.8209716081619263]
# 0.12480139950327196
# 시간 : 12.374280214309692


# LSTM
# Epoch 92: early stopping
# 194/194 [==============================] - 0s 1ms/step - loss: 0.3824 - mse: 0.3824 - mae: 0.4493
# 194/194 [==============================] - 0s 732us/step
# [0.3824264109134674, 0.3824264109134674, 0.4492783546447754]
# 0.7131595042000292
# 시간 : 29.465530157089233




