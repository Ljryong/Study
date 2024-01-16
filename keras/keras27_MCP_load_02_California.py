from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential , load_model
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1 데이터

datasets = fetch_california_housing()
print(datasets)

x = datasets.data
y = datasets.target

print(x.shape)      # (20640, 8)
print(y.shape)      # (20640,)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 0 , shuffle= True  )

from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler, StandardScaler, RobustScaler
# slr = MaxAbsScaler()
# slr = MinMaxScaler()
# slr = StandardScaler()
slr = RobustScaler()

slr.fit(x_train)
x_train = slr.transform(x_train)
x_test = slr.transform(x_test)


#2 모델구성
# model = Sequential()
# model.add(Dense(1,input_dim = 8))
# model.add(Dense(1))


# #3 컴파일, 훈련
# model.compile(loss = 'mse' , optimizer= 'adam', metrics=['mse','mae'] )

# from keras.callbacks import EarlyStopping , ModelCheckpoint
# es = EarlyStopping(monitor='val_loss' , mode = 'min' , patience= 10 , verbose = 1 , restore_best_weights=True )
# mpc = ModelCheckpoint(monitor = 'val_loss', mode = 'min' , verbose= 1 ,save_best_only=True , filepath = 'c:/_data/_save/MCP/keras26_MCP2.hdf5' )

# hist = model.fit(x_train,y_train, epochs= 100 ,batch_size= 100, validation_split=0.2 , callbacks = [es,mpc])
model = load_model('c:/_data/_save/MCP/keras26_MCP2.hdf5')

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
