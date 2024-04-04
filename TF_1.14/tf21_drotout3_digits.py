import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
import pandas as pd
from sklearn.preprocessing import MinMaxScaler , StandardScaler
tf.set_random_seed(777)

#1. 데이터
x,y = load_digits(return_X_y=True)


print(x.shape)
print(np.unique(y))

y = pd.get_dummies(y)

sc =MinMaxScaler()
x = sc.fit_transform(x)

xp = tf.compat.v1.placeholder(tf.float32, shape= [None,64])
yp = tf.compat.v1.placeholder(tf.float32, shape= [None,10])

w1 = tf.compat.v1.Variable( tf.compat.v1.random_normal([64,30]) ,dtype=tf.float32 , name='weight'  )
b1 = tf.compat.v1.Variable( tf.compat.v1.zeros([30]) ,dtype=tf.float32 , name='bias'  )
layer1 = tf.compat.v1.matmul(xp,w1) + b1         # 첫번째 레이어의 아웃 풋 = (None,10)

# model.add(Dense(9))
w2 = tf.compat.v1.Variable(tf.random_normal([30,40]) , name = 'weight2' )      # 아웃풋을 9로 빼기 위해서
b2 = tf.compat.v1.Variable(tf.zeros([40]) , name = 'bias2' )               
layer2 = tf.nn.softmax(tf.compat.v1.matmul(layer1,w2) + b2  )

# model.add(Dense(8))
w3 = tf.compat.v1.Variable(tf.random_normal([40,20]) , name = 'weight3' ) 
b3 = tf.compat.v1.Variable(tf.zeros([20]) , name = 'bias3' )              
layer3 = tf.nn.softmax(tf.compat.v1.matmul(layer2,w3) + b3 )                                   # 아웃풋은 (None,8)

keep_prob = tf.compat.v1.placeholder(tf.float32)

layer3 = tf.nn.dropout(layer3 , keep_prob= keep_prob)

# model.add(Dense(7))
w4 = tf.compat.v1.Variable(tf.random_normal([20,30]) , name = 'weight4' ) 
b4 = tf.compat.v1.Variable(tf.zeros([30]) , name = 'bias4' )              
layer4 = tf.compat.v1.matmul(layer3,w4) + b4                              

# model.add(Dense(1 , activation = 'sigmoid))
w5 = tf.compat.v1.Variable(tf.random_normal([30,10]) , name = 'weight5' ) 
b5 = tf.compat.v1.Variable(tf.zeros([10]) , name = 'bias5' ) 
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4,w5) + b5)

#3-1 컴파일
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y) )
loss = tf.reduce_mean(-tf.reduce_sum(yp * tf.log(hypothesis), axis = 1))

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)
# train = optimizer.minimize(loss)

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.0001).minimize(loss)
# 위의 주석된 두줄과 같다

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(10001) : 
    _,loss_v  = sess.run([train , loss ], feed_dict={xp : x , yp : y , keep_prob : 0.5 })
    if step % 20==0:
        print(step ,'\t' , loss_v,'\t'  )

from sklearn.metrics import accuracy_score
# predict = tf.cast(hypothesis > 0.5 , dtype=tf.float32)

y_predict = sess.run(hypothesis , feed_dict={xp : x ,  keep_prob : 1.0 })
print(y_predict)        # (8,3)의 데이터
# y_predict = sess.run(tf.argmax(y_predict,1))
# print(y_predict)          # [2 0 0 0 2 0 2 2]
y_predict = np.argmax(y_predict,axis=1)
print(y_predict)            # [2 0 0 0 2 0 2 2]

x = np.array(x)
y = np.array(y)

y_data = np.argmax(y,axis=1)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data,y_predict)
print('acc : ',acc)

# acc :  0.9065108514190318