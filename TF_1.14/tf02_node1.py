import tensorflow as tf

# 3 + 4 = ?
node1 = tf.constant(3.0 , tf.float32 )      # == 3
node2 = tf.constant(4.0)      # == 4 , 명시하지 않아도 먹힘

node3 = node1 + node2
print(node3)        # Tensor("add:0", shape=(), dtype=float32)
node3 = tf.add(node1,node2)
print(node3)        # Tensor("Add_1:0", shape=(), dtype=float32)

sess = tf.Session()
print(sess.run(node3))  # 7.0