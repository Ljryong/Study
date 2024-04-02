import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits , fetch_covtype
import pandas as pd
from sklearn.preprocessing import MinMaxScaler , StandardScaler
tf.set_random_seed(777)

#1. 데이터
path = 'C:\_data\dacon\wine\\'

train = pd.read_csv(path + 'train.csv' , index_col=0)
test = pd.read_csv(path + 'test.csv' , index_col=0)

train['type'] = train['type'].replace({'white': 0, 'red':1})
test['type'] = test['type'].replace({'white': 0, 'red':1})

x = train.drop(['quality'],axis=1)
y = train['quality']

print(x.shape)
print(pd.value_counts(y))

y = pd.get_dummies(y)

sc = MinMaxScaler()
x = sc.fit_transform(x)

xp = tf.compat.v1.placeholder(tf.float32, shape= [None,12])
yp = tf.compat.v1.placeholder(tf.float32, shape= [None,7])

w1 = tf.compat.v1.Variable(tf.random_normal([12,7]) , name = 'weight' )
b1 = tf.compat.v1.Variable(tf.zeros([7]) , name = 'bias' )             # 더 해지는 것이라서 행을 맞추지 않고 열을 아웃풋과 맞춰야 됨
layer1 = tf.nn.softmax(tf.compat.v1.matmul(xp,w1) + b1)

w2 = tf.compat.v1.Variable(tf.random_normal([7,7]) , name = 'weight' )
b2 = tf.compat.v1.Variable(tf.zeros([7]) , name = 'bias' )  
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer1,w2) + b2)


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
    _,loss_v = sess.run([train , loss ], feed_dict={xp : x , yp : y })
    if step % 20==0:
        print(step ,'\t' , loss_v,'\t'  )

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

# acc :  0.5062761506276151