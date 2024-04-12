import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)

#1. 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train) , (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#2. 모델구성
# Layer 1 
x = tf.compat.v1.placeholder(tf.float32 , shape=[None,28,28,1] )        # None 으로 두는 이유는 batch 로 자를수도 있고 다른 여러가지 이유로 None 으로 둠
y = tf.compat.v1.placeholder(tf.float32 , shape=[None,10] )

w1 = tf.compat.v1.get_variable('w1', shape=[2,2,1,128] )     # shape=[커널사이즈(2,2), 컬러(채널)(1) , 필터(아웃풋)(64)]
b1 = tf.compat.v1.Variable(tf.zeros([128]), name='bias' )
L1 = tf.nn.conv2d(x, w1, strides = [1,1,1,1] , padding = 'VALID' )              
L1 = L1 + b1
# L1 += b1 위의 식과 같다
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1,ksize=[1,2,2,1] ,strides = [1,2,2,1] , padding = 'VALID'  )    # keras 의 default 값
# striide = 4차원이라서 이렇게 사용하고 [ 1 , 2 , 2, 1 ] 가운데만 stride이고 맨앞과 맨 뒤는 shape를 맞춰주는 것이다
# 가운데 2개의 숫자를 가지고 strides 를 바꿀 수 있다 

# 위의 내용은 밑에 1줄과 똑같다
# model.add(Conv2d(64,kenel_size = (2,2) , stride = (1,1) ,input_shape = (28,28,1) ))

print(L1_maxpool)

""" print(w1)           # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1)           # Tensor("Conv2D:0", shape=(?, 27, 27, 64), dtype=float32)

# Layer 2
w2 = tf.compat.v1.get_variable('w2' , shape = [3,3, 64,32] )    # shape=[커널사이즈(3,3), 컬러(채널)(64) , 필터(아웃풋)(32)]
L2 = tf.nn.conv2d(L1 , w2 ,strides=[1,1,1,1] , padding = 'SAME' )   # Tensor("Conv2D_1:0", shape=(?, 27, 27, 32), dtype=float32)

print(L2) """