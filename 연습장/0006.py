# 주말 과제
# 파이썬 함수랑 클래스 차이점을 알고 정리해서 예를 포함하여 선생님한테 메일로 보내기
# 함수는 이거고 클래스는 이거란걸 설명하고 이해를 했다라는 과정을 보내기

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

#1
x = datasets.data
y = datasets.target

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state = 0)

#2
model = Sequential()
model.add(Dense(1,input_dim = 1))
model.add(Dense(1))

#3
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train,y_train,epochs = 1000 , batch_size = 1)
end_time = time.time()

#4
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

print(r2)
print(end_time - start_time)


# 시각화 (그래프 뽑아내기)
import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.plot(x,result , color = 'red')
plt.show()

