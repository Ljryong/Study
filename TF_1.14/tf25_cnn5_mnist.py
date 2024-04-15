import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)
# tf.compat.v1.enable_eager_execution()
tf.compat.v1.disable_eager_execution()  

#1. 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train) , (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

#2. 모델구성
# Layer 1 
x = tf.compat.v1.placeholder(tf.float32 , shape=[None,28,28,1] )        # None 으로 두는 이유는 batch 로 자를수도 있고 다른 여러가지 이유로 None 으로 둠
y = tf.compat.v1.placeholder(tf.float32 , shape=[None,10] )
drop = tf.compat.v1.placeholder(tf.float32 )

w1 = tf.compat.v1.get_variable('w1', shape=[2,2,1,32] )     # shape=[커널사이즈(2,2), 컬러(채널)(1) , 필터(아웃풋)(64)]
b1 = tf.compat.v1.Variable(tf.zeros([32]), name='bias1' )
L1 = tf.nn.conv2d(x, w1, strides = [1,1,1,1] , padding = 'VALID' )              
L1 = L1 + b1
# L1 += b1 위의 식과 같다
L1 = tf.nn.relu(L1)
L1 = tf.nn.dropout(L1 , rate=drop ) # keep_prob 는 0.8만큼 쓰겠다 0.2를 버리는것 keras 에서는 0.2를 쓰면 0.2를 버린다
                      # rate = 0.2 를 쓰면 keep_prob=0.8 이랑 같다
L1_maxpool = tf.nn.max_pool2d(L1,ksize=[1,2,2,1] ,strides = [1,2,2,1] , padding = 'VALID'  )    # keras 의 default 값
# striide = 4차원이라서 이렇게 사용하고 [ 1 , 2 , 2, 1 ] 가운데만 stride이고 맨앞과 맨 뒤는 shape를 맞춰주는 것이다
# 가운데 2개의 숫자를 가지고 strides 를 바꿀 수 있다 

# 위의 내용은 밑에 1줄과 똑같다
# model.add(Conv2d(64,kenel_size = (2,2) , stride = (1,1) ,input_shape = (28,28,1) ))

# Layer 2 
w2 = tf.compat.v1.get_variable('w2', shape=[3,3,32,16] )     # shape=[커널사이즈(2,2), 컬러(채널)(1) , 필터(아웃풋)(64)]
b2 = tf.compat.v1.Variable(tf.zeros([16]), name='bias2' )
L2 = tf.nn.conv2d(L1_maxpool, w2, strides = [1,1,1,1] , padding = 'SAME' )              
L2 = L2 + b2
# L2 += b1 위의 식과 같다
L2 = tf.nn.relu(L2)
L2 = tf.nn.dropout(L2 , rate=drop ) # keep_prob 는 0.8만큼 쓰겠다 0.2를 버리는것 keras 에서는 0.2를 쓰면 0.2를 버린다
                      # rate = 0.2 를 쓰면 keep_prob=0.8 이랑 같다
L2_maxpool = tf.nn.max_pool2d(L2,ksize=[1,2,2,1] ,strides = [1,2,2,1] , padding = 'VALID'  ) 


# Layer 3
w3 = tf.compat.v1.get_variable('w3', shape=[3,3,16,16] )     # shape=[커널사이즈(2,2), 컬러(채널)(1) , 필터(아웃풋)(64)]
b3 = tf.compat.v1.Variable(tf.zeros([16]), name='bias3' )
L3 = tf.nn.conv2d(L2_maxpool, w3, strides = [1,1,1,1] , padding = 'SAME' )              
L3 = L3 + b3
# L2 += b1 위의 식과 같다
L3 = tf.nn.elu(L3)
# L3 = tf.nn.dropout(L3 , keep_prob=0.9 ) # keep_prob 는 0.8만큼 쓰겠다 0.2를 버리는것 keras 에서는 0.2를 쓰면 0.2를 버린다
                      # rate = 0.2 를 쓰면 keep_prob=0.8 이랑 같다
# L3_maxpool = tf.nn.max_pool2d(L3,ksize=[1,2,2,1] ,strides = [1,2,2,1] , padding = 'VALID'  ) 

# print(L3)

# Flatten
L_flat = tf.reshape(L3 , [-1,6*6*16])
# print('flatten' , L_flat)

# Layer4 DNN
w4 = tf.compat.v1.get_variable('w4' , shape=[6*6*16 , 100] )
b4 = tf.compat.v1.Variable(tf.zeros([100]), name = 'b4')
L4 = tf.matmul(L_flat , w4) + b4
L4 = tf.nn.relu(L4)
L4 = tf.nn.dropout( L4, rate=drop)

# Layer5 DNN
w5 = tf.compat.v1.get_variable('w5' , shape=[100, 10] )
b5 = tf.compat.v1.Variable(tf.zeros([10]), name = 'b5')
L5 = tf.matmul(L4 , w5) + b5
hypothesis = tf.nn.softmax(L5)

#3 컴파일
# loss = tf.reduce_mean(-tf.reduce_sum(y*tf.compat.v1.log(hypothesis-y),axis=1))
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis + 1e-7 ),axis=1))


optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate= 0.001 )
train = optimizer.minimize(loss)

#3-2 훈련 , 4 평가, 예측
sess = tf.compat.v1.Session() 
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 1001
batch_size = 100
total_batch = int(len(x_train) / batch_size)            # 60000/100     int 를 붙인 이유는 숫자가 떨어지지 않을때를 방지


for step in range(epochs) : 
    
    avg_cost = 0
    for i in range(total_batch) : 
        start = i * batch_size
        end = start + batch_size
        
        batch_x , batch_y = x_train[start:end] , y_train[start:end]
        feed_dict = {x:batch_x , y :batch_y , drop : 0.3 }
    
        _, loss_val , w_v, b_v = sess.run([train , loss ,w5 , b5] , feed_dict =  feed_dict  )
        
        avg_cost += loss_val
        
        # i +=1 은 i = i+1 이랑 같다
        
    if step % 20==0:
        print(step , avg_cost )
        
from sklearn.metrics import accuracy_score
import numpy as np

y_predict = sess.run(hypothesis , feed_dict={x : x_test , y : y_test , drop : 1.0 })
print(y_predict)          
y_predict = np.argmax(y_predict,axis=1)
print(y_predict)            

y_data = np.argmax(y_test,axis=1)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data,y_predict)
print('acc : ',acc)

sess.close()