# 실습
# lr 수정해서 epoch 101번 이하로 줄여서
# step = 100 이하 , w = 1.99 , b = 0.99

import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)


#1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]

x = tf.compat.v1.placeholder(dtype=tf.float32, shape = [None] ) # x 와 y 값을 placeholder에 넣고 결과 뽑아보기 
y = tf.compat.v1.placeholder(dtype=tf.float32, shape = [None] ) 

# w = tf.Variable(111,dtype=tf.float32)
# b = tf.Variable(0,dtype=tf.float32)
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)    # random_normal  정규화 
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(w))


#2. 모델구성
# y = wx + b
# hypothesis = w * x + b        # hypothesis = y 
hypothesis = x * w + b

#3. 컴파일 , 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))        # mse // hypothesis(예측값) - y 를 해서 loss 를 알아냄

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08235421246584561)           # 경사감법
train = optimizer.minimize(loss)        # loss를 최소화하겠다
# model.compile(loss = 'mse', optimizer='sgb') 위에 3줄과 이거랑 똑같다

#3-2 훈련
# sess = tf.compat.v1.Session()

loss_val_list = []
w_val_list = []
b_val_list = []

with tf.compat.v1.Session() as sess :
    init = sess.run(tf.global_variables_initializer())

    # model.fit
    epochs = 101
    for step in range(epochs):      # for문 한번 돌 때마다 append 시켜서 loss 의 hist 를 기록한다
        """ sess.run(train, feed_dict={x: [1,2,3,4,5] , y : [3,5,7,9,11] } )  """
        _ , loss_val , w_val , b_val = sess.run([train, loss, w,b], feed_dict={x : x_data , y : y_data } )
        
        if step % 20 == 0:          # % 나머지 연산 // 20으로 나누어 떨어질 때 조건을 만족하는 것을 출력 // verbose랑 똑같음
            print(step,loss_val , w_val , b_val)        
            """ print(step,sess.run(loss,feed_dict={x: [1,2,3,4,5] , y : [3,5,7,9,11] } ),  # 여기에만 넣는 이유는 loss 를 구하는데에는 x,y 가 필요하다
                  sess.run(w ),     # w를 구하는 데에는 x,y 가 쓰이지 않아서 쓰지 않아도 됨
                  sess.run(b ) )    # b를 구하는 데에는 x,y 가 쓰이지 않아서 쓰지 않아도 됨 """
                # """"""가 있는 애들끼리 묵어서 생각해야 됨
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
        b_val_list.append(b_val)
        
        
    # 실습
    x_pred = [6,7,8] 

    # 뽑기 
    x_test = tf.compat.v1.placeholder(tf.float32 , shape = [None])

    # 실습
    x_pred_data = [6,7,8] 

    # 뽑기 
    x_test = tf.compat.v1.placeholder(tf.float32 , shape = [None])

    # 파이썬 방식
    y_predict = x_pred_data * w_val + b_val
    print('[6,7,8]의 예측 : ' , y_predict )    # model.predict 하는 법

    # placeholder에 넣어서 하는 방식 
    y_predict = x_test * w_val + b_val
    print('[6,7,8]의 예측 : ' , sess.run(y_predict , feed_dict = {x_test : x_pred_data }) )     # model.predict 하는 법


    a678 = sess.run(hypothesis , feed_dict = {x : x_pred } )    # model.predict 하는 법

    print('[6,7,8]의 예측 : ' , a678 )
    
    print('loss hist : ',loss_val_list )
    print('w hist : ',w_val_list)
    print('b hist : ',b_val_list)
    
    plt.figure(figsize=(7,7))
    plt.subplot(2,2,1)                      # 밑에 쓰면 안되고 항상 위에 써야 됨
    plt.plot(loss_val_list, c = 'red')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    
    plt.subplot(2,2,2)
    plt.plot(w_val_list, c = 'green' )
    plt.xlabel('epochs')
    plt.ylabel('w')

    plt.subplot(2,2,3)
    plt.plot(b_val_list, c = 'blue' )
    plt.xlabel('epochs')
    plt.ylabel('b')

    plt.subplot(2,2,4)
    plt.scatter(w_val_list,loss_val_list)
    plt.xlabel('w')
    plt.ylabel('loss')
    
    
    plt.show()
    
    
