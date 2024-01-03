import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1 데이터
x = np.array([range(10)])
print(x)
print(x.shape)
x = x.transpose()
print(x)
print(x.shape)                          # (10,3)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              [9,8,7,6,5,4,3,2,1,0]])

y = y.T

#2 모델구성
model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(3))

#3 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=700,batch_size = 1)

#4 평가, 예측
loss = model.evaluate(x,y)
result = model.predict([10])
print(loss)
print(result)

