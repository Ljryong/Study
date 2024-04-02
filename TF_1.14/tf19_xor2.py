import tensorflow as tf
tf.compat.v1.set_random_seed(777)

#1 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]          # (4,2)
y_data = [[0],[1],[1],[0]]


#2 모델
# model.add(Dense(10,input_dim = 2))
x = tf.compat.v1.placeholder(tf.float32 , shape = [None , 2] )
y = tf.compat.v1.placeholder(tf.float32 , shape = [None , 1] )

w1 = tf.compat.v1.Variable(tf.random_normal([2,10]) , name = 'weight1' )      # (2,10) 으로 해가지고 인풋 2 아웃풋 10인 레이어 형성
b1 = tf.compat.v1.Variable(tf.zeros([10]) , name = 'bias1' )                   # 아웃풋이 10개 이므로 bias 도 10
layer1 = tf.compat.v1.matmul(x,w1) + b1         # 첫번째 레이어의 아웃 풋 = (None,10)

# model.add(Dense(9))
w2 = tf.compat.v1.Variable(tf.random_normal([10,9]) , name = 'weight2' )      # 아웃풋을 9로 빼기 위해서
b2 = tf.compat.v1.Variable(tf.zeros([9]) , name = 'bias2' )               
layer2 = tf.sigmoid(tf.compat.v1.matmul(layer1,w2) + b2  )

# model.add(Dense(8))
w3 = tf.compat.v1.Variable(tf.random_normal([9,8]) , name = 'weight3' ) 
b3 = tf.compat.v1.Variable(tf.zeros([8]) , name = 'bias3' )              
layer3 = tf.compat.v1.matmul(layer2,w3) + b3                                    # 아웃풋은 (None,8)

# model.add(Dense(7))
w4 = tf.compat.v1.Variable(tf.random_normal([8,7]) , name = 'weight4' ) 
b4 = tf.compat.v1.Variable(tf.zeros([7]) , name = 'bias4' )              
layer4 = tf.compat.v1.matmul(layer3,w4) + b4                              

# model.add(Dense(1 , activation = 'sigmoid))
w5 = tf.compat.v1.Variable(tf.random_normal([7,1]) , name = 'weight5' ) 
b5 = tf.compat.v1.Variable(tf.zeros([1]) , name = 'bias5' ) 
hypothesis = tf.sigmoid(tf.compat.v1.matmul(layer4,w5) + b5)

#3-1 컴파일
loss = -tf.reduce_mean(y*tf.log(hypothesis) + (1 - y)*tf.log(1-hypothesis)  )           # binary_crossentropy

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(loss)



#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(100001) : 
    _, loss_val  = sess.run([train , loss], feed_dict={x : x_data , y : y_data})
    if step % 20==0:
        print(step , '\t', loss_val, )
        
        
predict = tf.cast(hypothesis > 0.5 , dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predict , y) , dtype=tf.float32))
hypo, pred, acc = sess.run([hypothesis, predict, acc],feed_dict = {x:x_data, y:y_data})
print("훈련값 : ", hypo)
print("예측값 : ", pred)
print("정확도 : ", acc)

# y_predict = np.round(y_predict)

# acc = sess.run(acc , feed_dict={y : y_data})
sess.close()

print('acc : ', acc)

# 예측값 :  [[0.]
#  [1.]
#  [1.]
#  [0.]]
# 정확도 :  1.0
# acc :  1.0