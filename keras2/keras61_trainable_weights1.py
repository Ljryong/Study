import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
# tf.random.set_seed(777)
# np.random.seed(777)
print(tf.__version__)


#1 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2 모델
model = Sequential()
model.add(Dense(3,input_dim = 1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
# Total params: 17
# Trainable params: 17              훈련한 파라미터
# Non-trainable params: 0


print(model.weights)                                                       # 초기 가중치 // 커널 = 가중치 
# [<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.47288632, -0.78825045,  1.2209238 ]], dtype=float32)>,
# <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.](바이어스의 초기 값), dtype=float32)>,
# <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
# array([[-0.1723156 ,  0.5125139 ],
#        [ 0.41434443, -0.8537577 ],
#        [ 0.5188304 , -0.91461056]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>,
#        <tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=
# array([[ 1.1585606],
#        [-0.4251585]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

print('================================================================================')
print(model.trainable_weights)
print('================================================================================')
print(len(model.weights))                           # 6 모델의 가중치가 몇개 있는지 보여줌
print(len(model.trainable_weights))                 # 6

###############################################
model.trainable = False # 중요                      default는 True 값이다
# 훈련 자체를 안하겠다 = 가중치를 그대로 사용하겠다
# 훈련을 시키면 남의 가중치를 사용하지 않는 것이라서 
# 기존의 가중치가 날아가서 처음부터 되는거라 안좋아질 확률이 높음
###############################################

print('================================================================================')
print(len(model.weights))                           # 6
print(len(model.trainable_weights))                 # 0 // 훈련 자체를 안하겠다 = 가중치를 그대로 사용하겠다
                                                    # 훈련을 시키면 남의 가중치를 사용하지 않는 것이라서 
                                                    # 기존의 가중치가 날아가서 처음부터 되는거라 안좋아질 확률이 높음

print(model.weights)

print(model.trainable_weights)
# false 를 하고 뽑으면 아무것도 나오지 않음 [] 형태로 나옴

model.summary()
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17          훈련을 안한 파라미터

# 아예 훈련을 안할 일은 진짜 거의 없다 









