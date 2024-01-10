from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
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

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 0 , shuffle= True)

#2 모델구성
model = Sequential()
model.add(Dense(1,input_dim = 8))
model.add(Dense(1))


#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer= 'adam')

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss' , mode = 'min' , patience= 10 , verbose = 1 , restore_best_weights=True)

hist = model.fit(x_train,y_train, epochs= 10000 ,batch_size= 100, validation_split=0.2 , callbacks = [es])


#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)



plt.figure(figsize = (9,6))
plt.plot(hist.history['loss'], c = 'red' , label = 'loss' , marker = '.')
plt.plot(hist.history['val_loss'],c = 'blue' , label = 'val_loss' , marker = '.')
plt.legend(loc = 'upper right')


print(hist)
plt.title('california loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()

plt.show()


print(loss)
print(r2)



# Epoch 220: early stopping
# 194/194 [==============================] - 0s 368us/step - loss: 0.6333
# 194/194 [==============================] - 0s 349us/step
# 0.6333415508270264 = loss
# 0.5249593856949173 = r2