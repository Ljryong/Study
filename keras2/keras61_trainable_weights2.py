import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)


#1 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2 모델
model = Sequential()
model.add(Dense(3,input_dim = 1))
model.add(Dense(2))
model.add(Dense(1))

####################### 중요 ########################
model.trainable = False # 중요                      default는 True 값이다
# model.trainable = True # default는 True 값이다
# 훈련 자체를 안하겠다 = 가중치를 그대로 사용하겠다
# 훈련을 시키면 남의 가중치를 사용하지 않는 것이라서 
# 기존의 가중치가 날아가서 처음부터 되는거라 안좋아질 확률이 높음
#####################################################

print(model.weights)
print('================================================================================')

#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer='adam')
model.fit(x,y,batch_size=1 , epochs = 1000 , verbose = 0 )

#4 평가, 예측
y_predict = model.predict(x)
print(y_predict)




