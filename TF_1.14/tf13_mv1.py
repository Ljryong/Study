# mv 변수를 여러개 쓰겠다

import tensorflow as tf

#1. 데이터
x1_data = [73., 93., 89., 96., 73.,]        # 국어
x2_data = [80., 88., 91., 98., 66.,]        # 영어
x3_data = [75., 93., 90., 100., 70.,]        # 수학
y_data = [152., 185., 180., 196., 142.,]        # 환산점수

# 실습 맹그러!

x1 = tf.compat.v1.placeholder(tf.float32 , shape=[None])        # shape=[None]을 써주는 이유는 습관을 들이기 위해서 행렬연산에서 쓰게됨
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]) , dtype=tf.float32 , name='weights' )
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]) , dtype=tf.float32 , name='weights' )
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]) , dtype=tf.float32 , name='weights' )

b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]) , dtype=tf.float32 , name='weights' )


#2
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b 

#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))        # mse 
# loss = tf.reduce_mean(tf.absolute(hypothesis - y))        # mae 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)           # 경사감법
train = optimizer.minimize(loss)

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# w_hist = []
# loss_hist = []
# b_hist = []

for step in range(101) :
    _ , loss_v , w1_v,w2_v,w3_v,  b_v = sess.run( [train ,loss , w1,w2,w3,b] , feed_dict={x1 : x1_data ,x2 : x2_data,x3 : x3_data, y: y_data})
    print(step , '\t' , loss_v , '\t' , w1_v,'\t' , w2_v,'\t' , w3_v,'\t' , b_v)
    
    # w_hist.append(w_v)
    # b_hist.append(b_v)
    # loss_hist.append(loss_v)
    if step % 20 ==0:
        print(step,'loss : ' , loss_v)

y_predict = x1_data * w1_v + x2_data * w2_v +x3_data * w3_v + b_v

from sklearn.metrics import r2_score , mean_absolute_error

r2 = r2_score(y_data,y_predict )
mae = mean_absolute_error(y_data,y_predict)

sess.close()

print('r2 : ',r2)
print('mae : ',mae)