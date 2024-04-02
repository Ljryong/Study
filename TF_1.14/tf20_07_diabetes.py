import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 데이터 로드
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape)  # (442, 10) (442,)

y = y.reshape(-1,1)

# 데이터 전처리 및 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=777)

# 모델 구성
tf.compat.v1.reset_default_graph()  # 그래프 리셋

xp = tf.compat.v1.placeholder(tf.float64, shape=[None, 10])
yp = tf.compat.v1.placeholder(tf.float64, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 10], dtype=tf.float64))
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([10], dtype=tf.float64))
layer1 = tf.matmul(xp, w1) + b1

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 5], dtype=tf.float64))
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([5], dtype=tf.float64))
layer2 = tf.matmul(layer1, w2) + b2

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([5, 1], dtype=tf.float64))
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([1], dtype=tf.float64))
hypothesis = tf.matmul(layer2, w3) + b3

# 손실 함수
loss = tf.reduce_mean(tf.square(hypothesis - yp))

# 최적화 함수
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)

# 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    epochs = 100001
    for step in range(epochs):
        _, loss_val = sess.run([train, loss], feed_dict={xp: x_train, yp: y_train})

        if step % 10 == 0:
            print(step, 'Loss:', loss_val)

    # 테스트
    y_pred = sess.run(hypothesis, feed_dict={xp: x_test})

# 평가
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('R2 Score:', r2)
print('MSE:', mse)
