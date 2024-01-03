import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,6,5,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])         # 데이터를 30% 손해를 보더라도 해야되는 이유는 과접합을 조심하기 위해서이다.
y_train = np.array([1,2,3,4,6,5,7])         # 실무에서 보통 7 : 3 으로 나눈다. 7은 훈련 3은 평가로 사용한다. 

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2 모델구성
model = Sequential()
model.add(Dense(7, input_dim = 1))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x_train,y_train,epochs=1000 , batch_size= 2)

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict([110000,7])
print("로스 : " ,loss)
print("11000의 예측 값 : ",results)

# 8/8 [==============================] - 0s 2ms/step - loss: 0.2540 ----------- 훈련 로스 값
# 1/1 [==============================] - 0s 67ms/step - loss: 0.0157 ---------- 평가 로스 값
# 1/1 [==============================] - 0s 70ms/step
# 로스 :  0.01574091799557209 ------------------------------------------------- 평가 로스 값   //   훈련 로스 값과 평가 로스 값이 차이가 없는것이 좋은것이다.
# 좋다는 뜻은 성능이 좋다는것이 아니라 과적합 되지 않았다는 뜻이다.
# 11000, 7 예측 값 :  [[1.0658384e+05] = [10658.8384] , [6.9392419e+00]]



