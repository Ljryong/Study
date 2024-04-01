import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

#1. 데이터
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]
y_data = [[0,0,1],      # 2
          [0,0,1],
          [0,0,1],
          [0,1,0],      # 1
          [0,1,0],
          [0,1,0],
          [1,0,0],      # 0
          [1,0,0],]

x = tf.compat.v1.placeholder(tf.float32, shape= [None,4])
y = tf.compat.v1.placeholder(tf.float32, shape= [None,3])

w = tf.compat.v1.Variable(tf.random_normal([4,3]) , name = 'weight' )
b = tf.compat.v1.Variable(tf.zeros([1,3]) , name = 'bias' )             # 더 해지는 것이라서 행을 맞추지 않고 열을 아웃풋과 맞춰야 됨

#2 모델
hypothesis = tf.compat.v1.matmul(x,w) + b

#3-1 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y) )
# loss = tf.reduce_mean(tf.feature_column.categorical_column_with_vocabulary_list((1-y)*tf.log(y)) )

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)
# train = optimizer.minimize(loss)

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)
# 위의 주석된 두줄과 같다

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(101) : 
    _,loss_v , w_v , b_v = sess.run([train , loss , w, b], feed_dict={x:x_data, y : y_data })
    if step % 20==0:
        print(step ,'\t' , loss_v,'\t' , w_v,'\t' , b_v, )


