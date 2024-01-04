import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


#1 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])

x_train, x_test, y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state=16, shuffle=True)

#2 모델 구성
model = Sequential()
model.add(Dense(5,input_dim = 1))
model.add(Dense(3))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs = 1 , batch_size = 1)

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
result = model.predict(x)
print(loss)
print(result)


import matplotlib.pyplot as plt             # 그림을 뽑아낼 때 필요한 정보를 가져온다.

plt.scatter(x,y)                            # x, y를 흩뿌린다. scatter = 점을 찍는다.
plt.plot(x,result,color='red')              # x 랑 result 값을 가지고 빨간색선으로 그래프를 그려라. plot = 선을 그린다.
plt.show()                                  # 그래프 그린것을 보여줘
                                            # 사진에 보이는 점은 실제 데이터, 빨간색 선은 예측 데이터이다.


# 14/14 [==============================] - 0s 1ms/step - loss: 9.9060
# 1/1 [==============================] - 0s 73ms/step - loss: 25.5197
# 1/1 [==============================] - 0s 33ms/step
# 25.51967430114746
# [[ 2.1473734]
#  [ 2.8505573]
#  [ 3.5537412]
#  [ 4.2569246]
#  [ 4.9601088]
#  [ 5.663292 ]
#  [ 6.3664756]
#  [ 7.06966  ]
#  [ 7.7728443]
#  [ 8.4760275]
#  [ 9.179213 ]
#  [ 9.882395 ]
#  [10.58558  ]
#  [11.288762 ]
#  [11.991948 ]
#  [12.69513  ]
#  [13.398314 ]
#  [14.101498 ]
#  [14.804681 ]
#  [15.507865 ]]