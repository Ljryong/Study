import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1 데이터
x = np.array([range(1,10)])               # numpy = 가져와서 쓰는거 // range = python 에서 기본적으로 제공하는 함수
                                        # range(a) = 괄호안에 숫자는 열이고 0 부터 시작된다.  
                                        # ex) range(10) = [[0,1,2,3,4,5,6,7,8,9]] = (1,10)
                                        # range(1,10) = [[1,2,3,4,5,6,7,8,9]]  앞에 숫자부터 뒤에 숫자 -1 만큼

print(x)                                # range(1,10) = [[1,2,3,4,5,6,7,8,9]]
print(x.shape)                          # (1,9)

x = np.array([range(10),range(21,31),range(201,211)])
print(x)                                # [[  0   1   2   3   4   5   6   7   8   9]
                                        #  [ 21  22  23  24  25  26  27  28  29  30]
                                        #  [201 202 203 204 205 206 207 208 209 210]]

print(x.shape)                          # (3, 10)
x = x.transpose()
print(x)
print(x.shape)                          # (10,3)

y = np.array([[1,2,3,4,5,6,7,8,9,10],[1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]])
#python에서 [] 안에 있는것을 "두개 이상은" list 라고 한다

print(y.shape)                          # (2, 10) / x.shape = (10,3) 이므로 y도 맞춰줘야한다.
                                        # 그래야 결과값을 도출할 수 있다.

y = y.T                                 # (10, 2)

#2 모델구성
model = Sequential()
model.add(Dense(5,input_dim = 3))
model.add(Dense(2))

#3 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y, epochs = 1000, batch_size = 1)

#4 예측, 평가       [10,31,211]
loss = model.evaluate(x,y)
result = model.predict([[10,31,211]])
print(loss)
print(result)


# 1/1 [==============================] - 0s 53ms/step
# 3.603535178586803e-11
# [[11.000002   1.9999932]]
# 52 epochs = 1000