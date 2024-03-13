import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())

tf.compat.v1.disable_eager_execution()

# node1 = tf.constant(30.0,tf.float32)
# node2 = tf.constant(40.0)
# node3 = tf.add(node1,node2)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)
add_node = a + b

print(sess.run(add_node, feed_dict={a:3, b : 4} ))      # 어떤 값을 넣든 add_node 를 실행
# 7.0
print(sess.run(add_node, feed_dict={a:30, b : 4.5} )) 
# 34.5

add_and_triple = add_node * 3
print(add_and_triple)

print(sess.run(add_and_triple, feed_dict={a:3,b:4} ))   # feed_dict를 쓰지 않으면 error 새로운 Session을 만들어주는거라고 생각해야됨  
# 21.0