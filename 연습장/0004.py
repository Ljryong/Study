import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1 데이터
x = np.array(range(1,11))
y = np.array([1,2,3,4,5,7,6,8,9,10])

x_train = x[:8]
y_train = y[:8]

x_test =x[8:]
y_test =y[8:]

print(x_train)
print(y_train)
print(x_test)
print(y_test)

#2 모델구성
model = Sequential()
model.add(Dense(5,input_dim = 1))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

#3 컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train , y_train, epochs = 1000, batch_size = 4 )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
result = model.predict([12500,9])

print("로스는 : ", loss)
print("결과는 : ", result)

# 과적합
# 8/8 [==============================] - 0s 2ms/step - loss: 0.2540
# 1/1 [==============================] - 0s 67ms/step - loss: 0.0158
# 1/1 [==============================] - 0s 48ms/step
# 로스는 :  0.015801550820469856s
# 결과는 :  [[1.218285e+04]
#  [8.887629e+00]]
# 로스값의 차이는 0.24


# 과적합이 아닐 때
# 2/2 [==============================] - 0s 0s/step - loss: 1.0786e-11
# 1/1 [==============================] - 0s 82ms/step - loss: 3.8654e-11
# 1/1 [==============================] - 0s 49ms/step
# 로스는 :  3.865352482534945e-11
# 결과는 :  [[1.2499984e+04]
#  [8.9999943e+00]]
# 로스값의 차이는 약 0.000000000002 정도

# 과적합으로 나타는 로스값의 차이는 매우 크다

#  31일

# git commit --amend --no-edit --date "Mon 20 Aug 2018 20:19:19 KST"