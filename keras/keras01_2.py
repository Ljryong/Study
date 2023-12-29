from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2 모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))

#3 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=15000)

#4 평가, 예측
loss = model.evaluate(x,y)
print("loss =",loss)
result = model.predict([1,2,3,4,5,6,7])
print("결과는" , result)

# loss = 0.3238096535205841
# 1/1 [==============================] - 0s 50ms/step
# 결과는 [[6.8001595]]
# 에포(epochs) : 15000


# loss = 0.3238094747066498
# 1/1 [==============================] - 0s 57ms/step
# 결과는 [[1.1428571]
#  [2.0857143]
#  [3.0285714]
#  [3.9714286]
#  [4.9142857]
#  [5.857143 ]
#  [6.8      ]]
# 에포(epochs) : 10000
