1+1
2+1

a = 1
b = 2
a + b
print(a+b)

c = a * b
print(c)






from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x = np.array([1,2,3])
y = np.array([1,2,3])

model = Sequential()
model.add(Dense(1,input_dim = 1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=2000)

loss=model.evaluate(x,y)
print(loss)
result=model.predict([4])
print(result)
