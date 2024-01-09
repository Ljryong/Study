from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

#1 데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)      # x = (20640, 8) ,  y =(20640,)
print(datasets.feature_names)       # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state= 59 , shuffle= True)

#2 모델구성
model = Sequential()
model.add(Dense(13,input_dim = 8))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(14))
model.add(Dense(7))
model.add(Dense(1))


#3 컴파일, 훈련
model.compile(loss= 'mae' , optimizer = 'adam')
model.fit(x_train, y_train , epochs = 3000 , batch_size = 100 , validation_split= 0.2)

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test , y_predict)

print(loss)
print(r2)

# Epoch 3000/3000
# 116/116 [==============================] - 0s 910us/step - loss: 0.5464 - val_loss: 0.5387
# 194/194 [==============================] - 0s 413us/step - loss: 0.5290
# 194/194 [==============================] - 0s 402us/step
# 0.5289812088012695 = loss
# 0.5582369514827683 = r2s