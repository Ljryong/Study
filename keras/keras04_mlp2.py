import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
             [9,8,7,6,5,4,3,2,1,0]])

y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x)
print(x.shape,y.shape)          # x.shape = (3,10) , y.shape = (10,)
                                # # print(x.shape , y.shape)이 가능한 이유는
                                # numpy에서 가져온 함수 때문에 가능한것 이고
                                # np.array를 사용하여 가져오는 것이다                           
x = x.T
print(x.shape)                  # 전치된 x.shape = (10,3)

#2 모델구성
model = Sequential()
model.add(Dense(1,input_dim = 3))           # 열, 컬럼, 속성, 특성, 차원 = 2 // 
                                            #(행무시,열우선) <= 외우기  (행의 갯수, 열의 갯수)
#행무시, 열우선 이란 행은 무시하고 n으로 두고 최소 갯수만 세어 (n,2) , (n,3) 과 같은 형식으로 만든다

#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=7000)

#4 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10,1.3,0]])        
# x와 스칼라 or 벡터를 맞춰줘야 되기 때문에 괄호가 2개이다. 모르겠으면 #1의 x 값과 똑같이 맞추면 된다. 
# (10,1.3)은 (2,) 벡터라서 구하는 값이 아니고 [[10,1.3]]으로 해야 (1,2)로 행렬로 해야한다.

print("[10,1.3,0]의 예측 값 : " , results)
print("loss의 값 : ",loss)


# 1/1 [==============================] - 0s 64ms/step
# [10,1.3,0]의 예측 값 :  [[9.999921]]
# loss의 값 :  1.93606020104653e-09
# 551 epochs = 3000

# 1/1 [==============================] - 0s 54ms/step
# [10,1.3,0]의 예측 값 :  [[10.]]
# loss의 값 :  3.6095570434512003e-13
# 551 epochs = 7000