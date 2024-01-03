import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

#실습 넘파이 리스트의 슬라이싱 7:3으로 정리

x_train = x[:7]             # :7 은 7번째 스칼라를 포함하고 밑에 모두를 지칭하는 것
y_train = y[:7]

x_test = x[7:]              # 7: 은 7번째 스칼라를 포함하고 위에 모두를 지칭하는 것
y_test = y[7:]             # 2:4 은 2는 포함하지 않고 4는 포함한다.

print(x_train)
print(y_train)
print(x_test)
print(y_test)


#2 모델구성
model = Sequential()
model.add(Dense(7, input_dim = 1))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x_train,y_train,epochs=1000 , batch_size= 10)

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict([110000,7])
print("로스 : " ,loss)
print("11000과 7의 예측 값 : ",results)

# 1 [==============================] - 0s 70ms/step - loss: 0.0161
# 1/1 [==============================] - 0s 58ms/step
# 로스 :  0.016122587025165558
# 11000과 7의 예측 값 :  [[1.06529555e+05]
#  [6.93876791e+00]]