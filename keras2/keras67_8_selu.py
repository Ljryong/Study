# 음수를 다 버리기 아까워서 생김

import numpy as np
import matplotlib.pyplot as plt

# def selu(x, alpha = 1.6732632423543772848170429916717, scale = 1.0507009873554804934193349852946 ):
#     return np.where(x <= 0, scale * alpha * (np.exp(x) - 1 ), scale * x)

alpha = 1.6732632423543772848170429916717
lambda_param = 1.0507009873554804934193349852946
selu = lambda x: np.where(x > 0, lambda_param * x, lambda_param * alpha * (np.exp(x) - 1))

x = np.arange(-5,5,0.1)
y = selu(x)

plt.plot(x,y)
plt.grid()
plt.show()
