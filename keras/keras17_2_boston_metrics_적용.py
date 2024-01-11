from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


#1 데이터
datasets = load_boston()

x = datasets.data
y = datasets.target
print(x.shape,y.shape)      # (506, 13) (506,)

x_train , x_test , y_train , y_test =train_test_split(x,y,test_size = 0.3 , random_state= 12 , shuffle= True)

es = EarlyStopping(monitor="val_loss" , mode = 'min' , verbose= 1 , patience= 10 , restore_best_weights=True)

#2 모델구성
model = Sequential()
model.add(Dense(512, input_dim = 13 ))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer= 'adam' , metrics=['mse','mae'])
hist = model.fit(x_train,y_train,epochs = 1000000 , batch_size = 1 , verbose = 1 , validation_split=0.2, callbacks = [es])

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

print("R2 = ",r2)
print('loss = ',loss)

plt.figure(figsize = (9,6))
plt.title('boston loss')
plt.plot(hist.history['loss'], c='red', label = 'loss' , marker = '.' )
plt.plot(hist.history['val_loss'] , c = 'blue' , label = 'val_loss' , marker="." )
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')

plt.show()







