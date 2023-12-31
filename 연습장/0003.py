from keras.models import Sequential
from keras.layers import Dense
import numpy as np


#1 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2 모델구성
model = Sequential()
model.add(Dense(1,input_dim=1))

#3 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=2500,batch_size=1)

#4 평가, 예측
loss = model.evaluate(x,y)
print(loss)
result = model.predict([4])
print(result)

# 1.3642420526593924e-12
# 1/1 [==============================] - 0s 33ms/step
# [[4.000003]]