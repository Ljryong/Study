import numpy as np
import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1. 데이터
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]      # (6,2)
y_data = np.array([[0],[0],[0],[1],[1],[1]])      # (6,1)

y_data = y_data.reshape(-1,1)

########## 실 습 ##########

xp = tf.compat.v1.placeholder(dtype=tf.float32 , shape=[None,2])
yp = tf.compat.v1.placeholder(dtype=tf.float32 , shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]),dtype=tf.float32, name='weight' )
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]),dtype=tf.float32, name='bias' )

#2 모델
hypothesis = tf.compat.v1.matmul(xp,w) + b

#3-1 컴파일
loss = tf.reduce_mean(tf.compat.v1.square(hypothesis - yp))           # binary_crossentropy 로 바꿔줘야 됨

#3-2 훈련
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)    

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(101) : 
    _ , loss_val , w_val , b_val = sess.run([ train , loss , w , b] , feed_dict={xp : x_data , yp : y_data} )
    if step % 20==0 :
        print(step , '\t' , loss_val,'\t' , w_val,'\t' , b_val,)

print(w_val)
# [[0.77237725]
#  [0.6599549 ]]
print(type(w_val))      # <class 'numpy.ndarray'> // sess.run 을 타고 나오는 것은 넘파이형식을 가진다

# #4 평가
x_test = tf.compat.v1.placeholder(tf.float32 , shape = [None,2] )
y_pred = tf.compat.v1.matmul(x_test,w_val) + b_val
y_predict = sess.run(y_pred , feed_dict={xp : x_data , yp: y_data})

print(y_predict)

from sklearn.metrics import mean_squared_error , accuracy_score , r2_score

r2 = r2_score(y_data,y_predict)
acc = accuracy_score(y_data,y_predict)
mse = mean_squared_error(y_data,y_predict)

sess.close()


print('r2 : ', r2)
print('acc : ', acc)
print('mse : ', mse)


