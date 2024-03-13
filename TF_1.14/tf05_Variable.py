import tensorflow as tf
sess = tf.compat.v1.Session()

a = tf.Variable( [2], dtype=tf.float32 )
b = tf.Variable( [3], dtype=tf.float32 ) 

init = tf.compat.v1.global_variables_initializer()                 # 변수 초기화

sess.run(init)      # 초기화도 run으로 시켜줘야 됨

print(sess.run(a+b))


