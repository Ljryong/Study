import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])            # 이런 데이터가 문제가 생기는 이유는 훈련한 데이터로 평가를 해서 문제가 생기는 것이다.
y = np.array([1,2,3,4,6,5,7,8,9,10])            # 훈련 데이터로 평가를 하면 과적합 되어있는지 판단을 할 수 없어서 훈련 데이터와 평가 데이터를 분리하는 것이다.
                                                # 나누는 비율은 7:3 으로 나눈다.

#2 모델구성
model = Sequential()
model.add(Dense(7, input_dim = 1))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x,y,epochs=750 , batch_size= 2)
#batch_size = 일괄 처리하기 위해서 잘라서 돌리는 것(한번에 많은 양을 돌릴 수 없을 때 사용) 하지만 epochs의 양은 많아진다. 85% 이상이 batch를 썻을때 성능이 좋아지지만 15%로 나빠질 경우도 있다
#면접에서 많이 물어보는 내용

#4 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([110000,7])             # 평가 데이터는 어떤 값이든 상관 없다. 값이 잘 나오는지 확인하는 것이지 다른 것이 아니다.
print("로스 : " ,loss)                          # 평가 데이터를 뽑을 때는 범위를 유지하면서 마지막쪽만이나 앞부분쪽만 빼는 부분은 거의 없다.
print("11000의 예측 값 : ",results)


# 로스 :  0.20177987217903137
# 11000의 예측 값 :  [[1.0776412e+05]
#  [6.9201484e+00]]
# y = np.array([1,2,3,4,6,5,7,8,9,10])
# 7은 0.08만큼 틀어졌지만 11000은 224만큼 더 많이 틀어져있다.
# 7은 훈련 안에 들어가 있어서 오차 범위가 적지만, 11000처럼 숫자가 커질수록 오차 범위가 늘어난다.






# 로스 :  3.012701308043042e-13
# 11000의 예측 값 :  [[1.1000001e+05]
#  [6.9999990e+00]]
# y = np.array([1,2,3,4,5,6,7,8,9,10])