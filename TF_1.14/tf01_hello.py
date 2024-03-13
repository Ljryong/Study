import tensorflow as tf
# print(tf.__version__)         # 1.14.0
# print('텐서플로로 hello world')

hello = tf.constant('hello world')
print(hello)        # constanant 상태로 Session에 저장되어 있는걸 보여줌 // hello world 라고 바로 안나옴
# Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session() # Session을 정의
print(sess.run(hello))         # 세션에 있는걸 run(밖으로 빼내는 작업) 해서 hello 안에 있는걸 뽑아냄
# b'hello world'  b = binary

