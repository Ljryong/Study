from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2 모델구성
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100)                   # 01_1번과 같은 결과를 만들기 loss 값을 0.33 정도로맞춘다

#4 평가, 예측
loss = model.evaluate(x,y)
print("loss =",loss)
result = model.predict([1,2,3,4,5,6,7])
print("결과는" , result)

# loss = 0.3250783383846283
# 1/1 [==============================] - 0s 65ms/step
# 결과는 [[1.1882378]
#  [2.1105382]
#  [3.0328395]
#  [3.9551404]
#  [4.8774424]
#  [5.7997437]
#  [6.7220445]]