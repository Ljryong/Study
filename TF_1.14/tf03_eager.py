import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())     # False / executing_eagerly = 즉시 실행 모드

# 즉시 실행 모드 -> 텐서 1의 그래프 형태의 구성없이 자연스러운 파이썬 문법으로 실행시킨다.
tf.compat.v1.disable_eager_execution()    # False / Default 값 / 즉시 실행 모드를 끈다 / 텐서플로 1.0 문법
# tf.compat.v1.enable_eager_execution()     # True / 즉시 실행모드 켠다 / 텐서플로 2.0 사용 가능 문법

# compat.v1 버전이 맞지 않으면 사용해야 됨

# Default 값은 다 다르다 

print(tf.executing_eagerly())   # True

hello = tf.constant('Hello World')

sess = tf.Session()

print(sess.run(hello))

# 가상환경         즉시실행모드             사용가능
# 1.14.0          disable(디폴트)          가능     ★★★★
# 1.14.0          enable                   에러(불가능)
# 2.9.0           disable                  가능     ★★★★    tensorflow2 환경에서 tensor 1 코드를 쓸 일이 있을 경우 사용
# 2.9.0           enable (디폴트)          에러(불가능)

""" 
Tensor 1 은 '그래프연산' 모드
Tensor 2 는 '즉시실행' 모드

tf.compat.v1.enable_eager_execution() 즉시실행모드 켜
                        -> Tensor 2 의 디폴트

tf.compat.v1.disable_eager_execution() 즉시실행모드 꺼 
                                -> 그래프 연산모드로 돌아간다.
                                -> Tensor 1 코드를 쓸 수 있다

tf.executing_eagerly()  True면 즉시실행모드, -> Tensor 2 코드만 써야한다.
                        False면 그래프연산모드 -> Tensor 1 코드를 쓸 수 있다
"""