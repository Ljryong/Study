import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
tf.set_random_seed(777)

(x_train , y_train) , (x_test , y_test ) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape , y_train.shape)        # (60000, 28, 28) (60000, 10)
print(x_test.shape , y_test.shape)          # (10000, 28, 28) (10000, 10)

x_train = x_train.reshape(60000,28*28).astype('float32')/255            # 127.5로 나누면 정규화시킬 수 있다(Standard -1~1 사이)
x_test = x_test.reshape(10000,28*28).astype('float32')/255             # 255는 (Minmaxscaler 0~1 사이)

# [실습] 맹그러

#2 모델구성

x = tf.compat.v1.placeholder(tf.float32 , shape = [None,784])
y = tf.compat.v1.placeholder(tf.float32 , shape = [None,10])

w1 = tf.compat.v1.get_variable( 'weight1', shape=[784,128] , initializer=tf.contrib.layers.xavier_initializer() ) 
# Variable 과 get_variable 약간의 차이는 있는데 체감은 안됨/ 알필요 없음
# initializer=tf.contrib.layers.xavier_initializer() 가중치를 초기화 하는 것 // 좋을수도 잇고 나쁠수도 있는데 좋을때가 생각보다 많음
# Variabler 에서는 initializer 를 사용하지 못하고 get_variable 에서만 사용 가능하다
################################## initializer 안에서 if 문으로 1에포 말고는 초기화가 되지 않는다 ##################################
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([128]) , dtype=tf.float32 , name='bias1' )
layer1 = tf.nn.relu(tf.compat.v1.matmul(x,w1)+b1)

keep_prob = tf.compat.v1.placeholder(tf.float32)

layer1 = tf.compat.v1.nn.dropout(layer1 , 
                                  keep_prob= keep_prob,
                                #   rate=0.3
                                  )

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([128,32]) , dtype=tf.float32 , name='weight2' )
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([32]) , dtype=tf.float32 , name='bias2' )
layer2 = tf.nn.relu(tf.compat.v1.matmul(layer1,w2)+b2)
layer2 = tf.compat.v1.nn.dropout(layer2 , 
                                  keep_prob= keep_prob,
                                #   rate=0.3
                                  )

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32,64]) , dtype=tf.float32 , name='weight3' )
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([64]) , dtype=tf.float32 , name='bias3' )
layer3 = tf.nn.relu(tf.compat.v1.matmul(layer2,w3)+b3)
layer3 = tf.compat.v1.nn.dropout(layer3 , 
                                  keep_prob= keep_prob,
                                #   rate=0.3
                                  )

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64,32]) , dtype=tf.float32 , name='weight4' )
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([32]) , dtype=tf.float32 , name='bias4' )
layer4 = tf.nn.relu(tf.compat.v1.matmul(layer3,w4)+b4)
layer4 = tf.compat.v1.nn.dropout(layer4 , 
                                  keep_prob= keep_prob,
                                #   rate=0.3
                                  )

w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32,10]) , dtype=tf.float32 , name='weight4' )
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]) , dtype=tf.float32 , name='bias4' )
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer4,w5) + b5)

#3-1 컴파일
# loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis-y),axis=1))
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.compat.v1.nn.log_softmax(hypothesis), axis=1))
# loss = tf.compat.v1.losses.softmax_cross_entropy(y,hypothesis)

optimizer = tf.train.AdamOptimizer(learning_rate= 0.001 )
train = optimizer.minimize(loss)

#3-2 훈련 , 4 평가, 예측
sess = tf.compat.v1.Session() 
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101
batch_size = 100
total_batch = int(len(x_train) / batch_size)            # 60000/100     int 를 붙인 이유는 숫자가 떨어지지 않을때를 방지


for step in range(epochs) : 
    
    avg_cost = 0
    for i in range(total_batch) : 
        start = i * batch_size
        end = start + batch_size
        
        batch_x , batch_y = x_train[start:end] , y_train[start:end]
        feed_dict = {x:batch_x , y :batch_y , keep_prob : 0.3 }
    
        _, loss_val , w_v, b_v = sess.run([train , loss ,w4 , b4] , feed_dict =  feed_dict  )
        
        avg_cost += loss_val
        
        # i +=1 은 i = i+1 이랑 같다
        
    if step % 20==0:
        print(step , avg_cost )
        
from sklearn.metrics import accuracy_score
import numpy as np

y_predict = sess.run(hypothesis , feed_dict={x : x_test , y : y_test , keep_prob : 1.0 })
print(y_predict)          
y_predict = np.argmax(y_predict,axis=1)
print(y_predict)            

y_data = np.argmax(y_test,axis=1)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data,y_predict)
print('acc : ',acc)

sess.close()