# diabetes = 당뇨병
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense

#1
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape,y.shape)              # x= (442,10) y= (442,)

print(datasets.feature_names)       # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

x_train, x_test , y_train , y_test = train_test_split(x,y,test_size=0.3, random_state= 785 )   #9
#2
model = Sequential()
model.add(Dense(50,input_dim = 10 ))
model.add(Dense(25))
model.add(Dense(1))
#3
model.compile(loss='mse', optimizer = 'adam')
start_time = time.time()
model.fit(x_train,y_train,epochs = 282 , batch_size = 10)
end_time = time.time()

#4
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

print('r2 : ', r2)
print('time : ' , end_time - start_time)

'''
import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.plot(x,result,color = 'red')
plt.show()
'''

# R2 0.62 기준 이상


# loss='mse'
# model.add(Dense(50,input_dim = 10 ))
# model.add(Dense(32))
# model.add(Dense(1))
# epochs = 222 , batch_size = 10
# 31/31 [==============================] - 0s 557us/step - loss: 3232.0852
# 5/5 [==============================] - 0s 813us/step - loss: 2187.3333
# 5/5 [==============================] - 0s 472us/step
# r2 :  0.6041937816532561
# time :  3.5137951374053955

