# SiLu(Sigmoid-Weighted Linear Unit) = Swish

# 음수를 다 버리기 아까워서 생김

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)

def silu(x) :
    return x * (1 / (1+np.exp(-x)))     # x * sigmoid

y = silu(x)

plt.plot(x,y)
plt.grid()
plt.show()
