from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

#1 데이터

datasets = load_boston()
print(datasets)
x = datasets.data
y = datasets.target
print(x.shape)      # (506, 13)
print(y.shape)      # (506,)

print(datasets.feature_names)       # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] = 항목 이름
print(datasets.DESCR)               # 설명

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 156 , shuffle=True)


#2 모델 구성
model = Sequential()
model.add(Dense(20,input_dim = 13))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(14))
model.add(Dense(7))
model.add(Dense(1))


#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs = 500 ,batch_size= 1 , validation_split = 0.2 )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

print("R2 : " , r2)
print("loss : " , loss)


# Epoch 500/500
# 283/283 [==============================] - 0s 682us/step - loss: 31.0327 - val_loss: 17.1739
# 5/5 [==============================] - 0s 748us/step - loss: 18.4386
# 5/5 [==============================] - 0s 607us/step
# R2 :  0.7412015373972498
# loss :  18.438636779785156


