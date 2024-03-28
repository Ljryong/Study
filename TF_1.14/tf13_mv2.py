import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1. 데이터
x1_data = [[73, 51., 65.,],
           [92, 98, 11,],
           [89,31,33],
           [99,33,100],
           [17,66,79]]        # 국어
y_data = [[152], [185], [180], [205], [142],]        # 환산점수

# 실습 맹그러!

x = tf.compat.v1.placeholder(tf.float32 , shape=[None,3])
y = tf.compat.v1.placeholder(tf.float32 , shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3,1],name = 'weight') , dtype=tf.float32  )
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], name= 'bias') , dtype=tf.float32  )

#2 모델
# hypothesis = x * w + b
hypothesis = tf.compat.v1.matmul(x,w) + b       # 행렬 연산에서는 위의 식이 먹히지 않아서 matmul을 사용해줘야된다

#3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)

train = optimizer.minimize(loss)

#3-2 훈련

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(101) :
    _, loss_val , w_val , b_val = sess.run([train , loss , w , b] , feed_dict={x:x1_data , y : y_data})
    if step % 20==0:
        print(step ,'loss', loss_val)




#4 평가
pred1 = tf.compat.v1.matmul(x1_data,w_val) + b_val          # 함수를 이용한 식이라서 sess.run 을 해줘야 한다
pred = sess.run(pred1 , feed_dict={x : x1_data , y: y_data})

from sklearn.metrics import mean_absolute_error , r2_score

r2 = r2_score(y_data,pred)
mae = mean_absolute_error(y_data,pred)

sess.close()

print('r2 : ', r2)
print('mae : ', mae)