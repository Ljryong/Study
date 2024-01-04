import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],[9,8,7,6,5,4,3,2,1,0]])
y = np.array([[1,2,3,4,5,7,6,8,9,10],[9,8,6,7,5,4,3,2,1,0]])

x_train = x[:,-3:]      #질문 앞에 3개와 뒤에 3개를 뽑아내는 방법을 모르겠음.
                        # [[ 8  9 10],[ 2  1  0]] 로 출력되는데 [[1,2,3],[2,1,0]]을 뽑고싶음

y_train = y[:6,:3]

print(x_train)


'''
x_test = x[]
y_test = y[]


#2 모델구성
model = Sequential()
model.add(Dense(1,input_dim = 2))
model.add(Dense(2))

#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=1000,batch_size=3)

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
result = 
'''
