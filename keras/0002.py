from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1 데이터
x = np.array([1,3,5,7,9])
y = np.array([1,3,5,7,9])

#2 모델구성
model = Sequential()
model.add(Dense(1 , input_dim = 1))


#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=3000)

#4 평가, 예측
loss = model.evaluate(x,y)
print(loss)
result = model.predict([11])
print(result)
