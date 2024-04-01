import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits , fetch_covtype
import pandas as pd
from sklearn.preprocessing import MinMaxScaler , StandardScaler
tf.set_random_seed(777)

#1. 데이터
datasets = fetch_covtype()

x = datasets.data
y = datasets.target

print(x.shape)
print(pd.value_counts(y))

y = pd.get_dummies(y)

sc = MinMaxScaler()
x = sc.fit_transform(x)

xp = tf.compat.v1.placeholder(tf.float32, shape= [None,54])
yp = tf.compat.v1.placeholder(tf.float32, shape= [None,7])

w = tf.compat.v1.Variable(tf.random_normal([54,7]) , name = 'weight' )
b = tf.compat.v1.Variable(tf.zeros([1,7]) , name = 'bias' )             # 더 해지는 것이라서 행을 맞추지 않고 열을 아웃풋과 맞춰야 됨

#2 모델
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(xp,w) + b)

#3-1 컴파일
# loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - y) )
loss = tf.reduce_mean(-tf.reduce_sum(yp * tf.log(hypothesis), axis = 1))

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)
# train = optimizer.minimize(loss)

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1).minimize(loss)
# 위의 주석된 두줄과 같다

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(101) : 
    _,loss_v , w_v , b_v = sess.run([train , loss , w, b], feed_dict={xp : x , yp : y })
    if step % 20==0:
        print(step ,'\t' , loss_v,'\t' , w_v,'\t' , b_v, )

#4 평가, 예측
y_predict = sess.run(hypothesis , feed_dict={xp : x })
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

# acc :  0.629899210343332