from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import pandas as pd


#1 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train , x_test , y_train , y_test = train_test_split(x,y,train_size = 0.625 , shuffle=False , random_state= 10)

print(x_train,y_train)
print(x_test , y_test)


#2 모델구성
model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

# #3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs = 500 , batch_size = 1 , 
          validation_split= 0.3 ,          # validation_split = x_val , y_val 을 쓰지 않고 train 값에서 validation 값을 뽑아낸다.
          verbose = 1 )

# #4 평가,예측
loss = model.evaluate(x_test,y_test)
result = model.predict([14,15,16])
print(loss)
print(result)




















