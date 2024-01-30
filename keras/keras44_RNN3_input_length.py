import numpy as np
from keras.models import Sequential 
from keras.layers import Dense , SimpleRNN ,Dropout


#1 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])    # timestack = 3 으로 자름
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)      # (7,3) , (7,)

x = x.reshape(7,3,1)
print(x.shape)               # (7, 3, 1)



#2 모델구성
model = Sequential()
# model.add(SimpleRNN(units=32,input_shape = (3,1)))        # (행무시 , 열우선)  = 행인 7을 제외하고 (3,1)이 인풋
# input_shape = timesteps, features
# 3-D tensor with shape (batch_size, timesteps, features)
# output 을 units 라고 부르고 숫자만 올 시 생략이 가능하다.
model.add(SimpleRNN(units=10 ,input_shape = (3,1)))          # 위아래 3개는 동일하다.
model.add(SimpleRNN(10, input_length =3 , input_dim = 1))
model.add(SimpleRNN(10, input_dim = 1, input_length =3  ))      # 하지만 얘는 가독성이 떨어져서 위에처럼 쓰지만 써도는 된다.

model.add(Dense(64 ,activation='relu'))
model.add(Dense(16 ,activation='relu'))
model.add(Dense(8 ,activation='relu'))
model.add(Dense(1))


#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer='adam')
model.fit(x,y,epochs= 300, batch_size=1  )


#4 평가, 예측
result = model.evaluate(x,y)
print('loss = ' , result)
y_predict = np.array([8,9,10]).reshape(1,3,1)   
# ([8,9,10]) 으로 넣으면 (3,) 여서 차원이 달라서 에러가 뜸
# 그래서 차원을 맞춰주기 위해서 reshape를 해준다
y_predict = model.predict(y_predict)      
# x값의 (8,9,10)을 안쓴이유는 predict에 넣기 위해서

print('결과' , y_predict)


# Epoch 100/100
# 7/7 [==============================] - 0s 2ms/step - loss: 3.1778e-05
# 1/1 [==============================] - 0s 92ms/step - loss: 2.2247e-05
# loss =  2.224672607553657e-05
# 1/1 [==============================] - 0s 78ms/step
# 결과 [[11.047786]]

# Epoch 1000/1000
# 7/7 [==============================] - 0s 3ms/step - loss: 1.4056e-04
# 1/1 [==============================] - 0s 166ms/step - loss: 4.1258e-06
# loss =  4.12583585784887e-06
# 1/1 [==============================] - 0s 126ms/step
# 결과 [[10.995547]]

# Epoch 100/100
# 7/7 [==============================] - 0s 2ms/step - loss: 3.5916e-05
# 1/1 [==============================] - 0s 92ms/step - loss: 2.4653e-05
# loss =  2.465285979269538e-05
# 1/1 [==============================] - 0s 81ms/step
# 결과 [[11.001695]]



