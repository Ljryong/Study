import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

#1. 데이터
x_train = [1,2,3]
y_train = [1,2,3]
x_test = [4,5,6]
y_test = [4,5,6]


x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable( [10] ,dtype=tf.float32 , name='weight')
b = tf.compat.v1.Variable( [10] ,dtype=tf.float32 , name='weight')

#2. 모델
hypothesis = x * w # + b            # hypothesis = 예측된 y 값

#3-1. 컴파일 // model.compile(loss='mse' , optimizer = 'sgb')
loss = tf.reduce_mean(tf.square(hypothesis - y))        # mse 

############################ 옵티마이저 ############################
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08235421246584561)           # 경사감법
# train = optimizer.minimize(loss)     
lr = 0.1
gradient = tf.reduce_mean((x * w  - y) * x )      # (x * w + b - y) * x  = tf.square(hypothesis - y) 를 미분 한 값 숫자 2는 결국 비율로 나타내는 것이라서 안써도 똑같다.
# + b
descent = w - lr * gradient

update = w.assign(descent)                  # 이 4줄이 주석과 위의 주석 2줄과 같은 내용
############################ 옵티마이저 ############################

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

w_history = []
loss_history = []
b_history = []

for step in range(21) : 
    _ , loss_v , w_v  = sess.run([update,loss , w] , feed_dict={x : x_train , y: y_train}) # , b_v , + b
    print(step , '\t' , loss_v , '\t' , w_v, )   # '\t' , b_v
    
    w_history.append(w_v)
    loss_history.append(loss_v)
    # b_history.append(b_v)
    

# plt.plot(loss_history)
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.show()

############ 실습 ############
# r2, mae 만들어
from sklearn.metrics import r2_score , mean_absolute_error

# placeholder에 넣어서 하는 방식 
y_predict = x_test * w_v # + b_v

r2 = r2_score(y_test,y_predict )
mae = mean_absolute_error(y_test,y_predict)

sess.close()

print('r2',r2)
print('mae' , mae)