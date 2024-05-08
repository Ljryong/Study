# 음수를 다 버리기 아까워서 생김

import numpy as np
import matplotlib.pyplot as plt

a = 0.1

def leakyrelu(x) :
    return np.maximum(a*x,x)

x = np.arange(-5,5,0.1)
y = leakyrelu(x)

plt.plot(x,y)
plt.grid()
plt.show()
