import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

#실습 넘파이 리스트의 슬라이싱 7:3으로 정리

x_train = x[:7]             # :7 은 7번째 스칼라를 빼고 밑에 모두를 지칭하는 것, 데이터는 0번부터 존재한다.
y_train = y[:7]             # 

'''
a = b                       # a 라는 변수에 b 값을 넣어라
a == b                      # a 와 b 가 같다
'''

x_test = x[7:]              # 7: 은 7번째 스칼라를 빼고 위에 모두를 지칭하는 것, 데이터는 0번부터 존재한다.
y_test = y[7:]              # 2:4 은 2는 포함하지 않고 4는 포함한다. // [7:10] == [-3:] == [-3:10]

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
model.fit(x_train,y_train,epochs=250 , batch_size= 2)

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict([110000,7])
print("로스 : " ,loss)
print("11000과 7의 예측 값 : ",results)

# 1/1 [==============================] - 0s 67ms/step - loss: 0.0021
# 1/1 [==============================] - 0s 60ms/step
# 로스 :  0.0021343149710446596
# 11000과 7의 예측 값 :  [[1.07483414e+05] , [7.00350571e+00]]
# 7,6,10,4,1  epochs = 250 , batch = 2