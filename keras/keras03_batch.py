# from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import keras
print("tf 버전:" , tf.__version__)
print("keras 버전 :" , keras.__version__)

#1 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2 모델구성
model = Sequential()
model.add(Dense(7, input_dim = 1))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(1))


#3 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x,y,epochs=100 , batch_size= 3)
# batch_size = 일괄 처리하기 위해서 잘라서 돌리는 것(한번에 많은 양을 돌릴 수 없을 때 사용) 하지만 epochs의 양은 많아진다. 85% 이상이 batch를 썻을때 성능이 좋아지지만 15%로 나빠질 경우도 있다
# = 1 epochs에 몇번 잘라 돌리는지 // 위와 같은 경우는 1에포에 123 456 으로 돌아간다
# 면접에서 많이 물어보는 내용

#4 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([7])
print("로스 : " ,loss)
print("7의 예측 값 : ",results)



# loss = 0.3250783383846283

# 1/1 [==============================] - 0s 70ms/step
# 로스 :  0.324065238237381
# 7의 예측 값 :  [[6.781629]]
# 17641   batch = 3  epochs=100

# 1/1 [==============================] - 0s 49ms/step
# 로스 :  0.32390472292900085
# 7의 예측 값 :  [[6.8172174]]
# 75641 batch = 3 epochs=100
