# 음수를 다 버리기 아까워서 생김

import numpy as np
import matplotlib.pyplot as plt

a = 1

def elu(x) :
    return (x>0)*x + (x<=0)*(a*(np.exp(x)-1))

x = np.arange(-5,5,0.1)

y = elu(x)

plt.plot(x,y)
plt.grid()
plt.show()
