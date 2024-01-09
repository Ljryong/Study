from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import time

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)          # (442, 10) (442,)
print(datasets.feature_names)   #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 9 , shuffle= True )

#2 모델구성
model = Sequential()
model.add(Dense(50,input_dim = 10 ))
model.add(Dense(25))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer = 'adam')
model.fit(x_train,y_train,epochs = 222 , batch_size = 10, validation_split = 0.2 )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print(r2)
print(loss)

# Epoch 222/222
# 25/25 [==============================] - 0s 1ms/step - loss: 3036.3245 - val_loss: 4001.1067
# 5/5 [==============================] - 0s 747us/step - loss: 2165.4966
# 5/5 [==============================] - 0s 498us/step
# 0.6081451926560789
# 2165.49658203125