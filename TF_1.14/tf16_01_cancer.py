from sklearn.datasets import load_iris , load_breast_cancer
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , MinMaxScaler , MaxAbsScaler
tf.compat.v1.set_random_seed(730501)

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

w = tf.compat.v1.Variable( tf.compat.v1.random_normal([30,1]) ,dtype=tf.float32 , name='weight'  )
b = tf.compat.v1.Variable( tf.compat.v1.zeros([1]) ,dtype=tf.float32 , name='bias'  )

#2 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(xp,w) + b )

#3-1 컴파일
loss = tf.reduce_mean(yp*tf.log(hypothesis)+(1 - yp)*tf.log(1-hypothesis) )

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(101) : 
    _, loss_val , w_val , b_val = sess.run([train , loss , w, b], feed_dict={xp : x , yp : y})
    if step % 20==0:
        print(step , '\t' , loss_val,'\t' , w_val,'\t' , b_val )


#4 평가
x_test = tf.compat.v1.placeholder(dtype=tf.float32)

y_predict = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x_test,w_val) + b_val)

y_pred = sess.run(y_predict , feed_dict={x_test: x})

y_pred = np.round(y_pred)

acc = accuracy_score(y , y_pred )

print('acc : ',acc)

# acc :  0.8541300527240774