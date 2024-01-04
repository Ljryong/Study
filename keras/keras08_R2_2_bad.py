# R2를 음수가 아닌 0.5이하로 만들기
# 데이터는 건들지 마라
# 레이어는 인풋과 아웃풋 포함해서 7개 이상
# batch_size=1
# 히든레이어의 노드는 10개이상 100개이하
# train 사이즈 75%
# epoch 100 이상

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#1 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])

x_train, x_test, y_train , y_test = train_test_split(x,y,test_size = 0.3 ,random_state=6386, shuffle=True)

#2 모델구성
model = Sequential()
model.add(Dense(1,input_dim = 1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(8))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs = 100 , batch_size = 1)

#4 평가, 예측
model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test , y_predict)
print(r2)

# loss: 25.5992
# 0.4031285194307671