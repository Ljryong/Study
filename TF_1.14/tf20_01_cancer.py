from sklearn.datasets import load_iris , load_breast_cancer
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , MinMaxScaler , MaxAbsScaler
tf.compat.v1.set_random_seed(220118)

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

scaler = MaxAbsScaler()
x = scaler.fit_transform(x)

# x , y  = datasets.data , datasets.target  # 위에랑 똑같다

print(x,x.shape)        # (569, 30)

y = y.reshape(-1,1)     # (569, 1)

print(y,y.shape)

xp = tf.compat.v1.placeholder(dtype=tf.float32 , shape=[None,30])
yp = tf.compat.v1.placeholder(dtype=tf.float32 , shape=[None,1])

w1 = tf.compat.v1.Variable( tf.compat.v1.random_normal([30,10]) ,dtype=tf.float32 , name='weight'  )
b1 = tf.compat.v1.Variable( tf.compat.v1.zeros([10]) ,dtype=tf.float32 , name='bias'  )
layer1 = tf.compat.v1.matmul(xp,w1) + b1         # 첫번째 레이어의 아웃 풋 = (None,10)

# model.add(Dense(9))
w2 = tf.compat.v1.Variable(tf.random_normal([10,9]) , name = 'weight2' )      # 아웃풋을 9로 빼기 위해서
b2 = tf.compat.v1.Variable(tf.zeros([9]) , name = 'bias2' )               
layer2 = tf.sigmoid(tf.compat.v1.matmul(layer1,w2) + b2  )

# model.add(Dense(8))
w3 = tf.compat.v1.Variable(tf.random_normal([9,8]) , name = 'weight3' ) 
b3 = tf.compat.v1.Variable(tf.zeros([8]) , name = 'bias3' )              
layer3 = tf.sigmoid(tf.compat.v1.matmul(layer2,w3) + b3 )                                   # 아웃풋은 (None,8)

# model.add(Dense(7))
w4 = tf.compat.v1.Variable(tf.random_normal([8,7]) , name = 'weight4' ) 
b4 = tf.compat.v1.Variable(tf.zeros([7]) , name = 'bias4' )              
layer4 = tf.compat.v1.matmul(layer3,w4) + b4                              

# model.add(Dense(1 , activation = 'sigmoid))
w5 = tf.compat.v1.Variable(tf.random_normal([7,1]) , name = 'weight5' ) 
b5 = tf.compat.v1.Variable(tf.zeros([1]) , name = 'bias5' ) 
hypothesis = tf.sigmoid(tf.compat.v1.matmul(layer4,w5) + b5)

#3-1 컴파일
loss = tf.reduce_mean(yp*tf.log(hypothesis)+(1 - yp)*tf.log(1-hypothesis) )

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(1001) : 
    _, loss_val  = sess.run([train , loss ], feed_dict={xp : x , yp : y})
    if step % 20==0:
        print(step , '\t' , loss_val,'\t')


predict = tf.cast(hypothesis > 0.5 , dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predict , y) , dtype=tf.float32))
hypo, pred, acc = sess.run([hypothesis, predict, acc],feed_dict = {xp:x, yp:y})
print("훈련값 : ", hypo)
print("예측값 : ", pred)
print("정확도 : ", acc)

# acc :  0.8541300527240774

# 정확도 :  0.6274165