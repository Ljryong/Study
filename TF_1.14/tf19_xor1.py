import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]          # (4,2)
y_data = [[0],[1],[1],[0]]

x = tf.compat.v1.placeholder(tf.float32 , shape = [None , 2] )
y = tf.compat.v1.placeholder(tf.float32 , shape = [None , 1] )

w = tf.compat.v1.Variable(tf.random_normal([2,1]) , name = 'weight' )
b = tf.compat.v1.Variable(tf.zeros([1]) , name = 'bias' )

#[실습] 맹그러

#2 모델
hypothesis = tf.sigmoid(tf.compat.v1.matmul(x,w) + b)

#3-1 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1 - y)*tf.log(1-hypothesis)  )

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(101) : 
    _, loss_val , w_val , b_val = sess.run([train , loss, w, b], feed_dict={x : x_data , y : y_data})
    if step % 20==0:
        print(step , '\t', loss_val,'\t', w_val,'\t', b_val, )

from sklearn.metrics import accuracy_score

y_pred = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x,w_val) + b_val)
y_predict = sess.run(y_pred , feed_dict={x : x_data})

import numpy as np

y_predict = np.round(y_predict)
acc = accuracy_score(y_data,y_predict)
sess.close()

print('acc : ', acc)
