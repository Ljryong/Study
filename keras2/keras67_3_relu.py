import numpy as np
import matplotlib.pyplot as plt

# activation 을 활성화 함수라고도 부르지만 한정 함수라고도 부른다 아웃풋의 값을 음수는 0으로 양수는 x값으로 한정하기 때문에

x = np.arange(-5,5,0.1)

# def relu(x) :
#     return np.maximum(0,x)

relu = lambda x : np.maximum(0,x)

y = relu(x)

plt.plot(x,y)
plt.grid()
plt.show()
# 난정말 시그모이드
