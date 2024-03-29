import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

#1. 데이터
x_train = [1,2,3]
y_train = [1,2,3]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable( [10] ,dtype=tf.float32 , name='weight')

#2. 모델
hypothesis = x * w            # hypothesis = 예측된 y 값

#3-1. 컴파일 // model.compile(loss='mse' , optimizer = 'sgb')
loss = tf.reduce_mean(tf.square(hypothesis - y))        # mse 

############################ 옵티마이저 ############################
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08235421246584561)           # 경사감법
# train = optimizer.minimize(loss)     
lr = 0.1
gradient = tf.reduce_mean((x * w - y) *x )

descent = w - lr * gradient

update = w.assign(descent)                  # 이 4줄이 주석과 위의 주석 2줄과 같은 내용
############################ 옵티마이저 ############################

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

w_history = []
loss_history = []

for step in range(21) : 
    _ , loss_v , w_v = sess.run([update,loss , w] , feed_dict={x : x_train , y: y_train})
    print(step , '\t' , loss_v , '\t' , w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)
sess.close()

plt.plot(loss_history)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()