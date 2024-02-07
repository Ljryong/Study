import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras
import random as rn

print(tf.__version__)
print(keras.__version__)
print(np.__version__)
rn.seed(333)
tf.random.set_seed(123)             # tensorflow seed 값 고정 시키는 법 // tensorflow 버전 2.9(먹힘) 2.15(안먹힘)
np.random.seed(321)                 # numpy seed 값 고정 시키는 법 / 가중치는 고정을 안시키는게 정설
                                    # 가중치를 고정 시킬라면 layer 마다 kernel_initializer='zeros' 를 써줘야된다.
                                    # 혹은 rrn.seed(123) // tf.random.set_seed(123) // np.random.seed(123)     3개 전부를 써주면 된다.

#1 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2 모델
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(5))
model.add(Dense(1)) # kernel_initializer='zeros'

#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer='adam')
model.fit(x,y,epochs=100)

#4 평가, 예측
loss= model.evaluate(x,y,verbose=0)
print('loss :',loss)
results = model.predict([4],verbose=0)
print('4의 예측값 :',results)









