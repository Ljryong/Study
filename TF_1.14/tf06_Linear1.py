import tensorflow as tf
tf.set_random_seed(777)


#1. 데이터
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(111,dtype=tf.float32)
b = tf.Variable(0,dtype=tf.float32)

#2. 모델구성
# y = wx + b
# hypothesis = w * x + b        # hypothesis = y 
hypothesis = x * w + b

#3. 컴파일 , 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))        # mse // hypothesis(예측값) - y 를 해서 loss 를 알아냄

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)           # 경사감법
train = optimizer.minimize(loss)        # loss를 최소화하겠다
# model.compile(loss = 'mse', optimizer='sgb') 위에 3줄과 이거랑 똑같다

#3-2 훈련
sess = tf.compat.v1.Session()
init = sess.run(tf.global_variables_initializer())

# model.fit
epochs = 100
for step in range(epochs):
    sess.run(train)             # 이게 훈련을 시키는 거고 밑에는 뽑아내는 것
    if step % 20 == 0:          # % 나머지 연산 // 20으로 나누어 떨어질 때 조건을 만족하는 것을 출력 // verbose랑 똑같음
        print(step,sess.run(loss),sess.run(w),sess.run(b) )     # verbose 와 model.weight 에서 봤던 애들

sess.close()        # sess가 남아있으면 메모리를 차지 함 닫아줘야 됨