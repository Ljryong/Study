import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# pip install numpy 를 했어야 됬다.

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

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              [9,8,7,6,5,4,3,2,1,0]])

y = y.T

#2 모델구성
model =Sequential()
model.add(Dense(3,input_dim = 3))



#3 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')
model.fit(x,y,epochs = 1000 , batch_size = 1)

#4 평가, 예측
loss = model.evaluate(x,y)
result = model.predict([[10,31,211]])
print(loss)
print(result)


# 1/1 [==============================] - 0s 47ms/step
# 7.050739853076138e-11
# [[10.999994   1.9999845 -0.9999788]]
# 53, epochs = 1000, batch = 1

# 1/1 [==============================] - 0s 48ms/step
# 1.0906949822475642e-11
# [[10.999998   1.9999982 -1.0000046]]
# 33, epochs = 2000, batch = 1

# 1/1 [==============================] - 0s 56ms/step
# 5.035579988543759e-07
# [[11.002438   2.0000021 -1.0000099]]
# 3, epochs = 1000, batch = 1