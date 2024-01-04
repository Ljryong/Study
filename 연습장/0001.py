from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1 데이터
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

#2 모델구성
model = Sequential()
model.add(Dense(1,input_dim =1 ))

#3 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs = 8000)


#4 평가, 예측
loss = model.evaluate(x,y)
print(loss)
result = model.predict([5])
print(result)

# 19.395360946655273
# 1/1 [==============================] - 0s 49ms/step
# [[-4.2249393]]

# 3.3431035717512714e-12
# 1/1 [==============================] - 0s 46ms/step
# [[4.999996]]