import numpy as np
from keras.models import Sequential
from keras.layers import Dense , LSTM , SimpleRNN
from keras.callbacks import EarlyStopping



# 1 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]]) 
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

print(x.shape)      # (13, 3)
x = x.reshape(13,3,1)

es = EarlyStopping(monitor='loss' , mode = 'min' , patience= 50 , restore_best_weights=True , verbose= 1 )

#2 모델구성
model = Sequential()
model.add(LSTM(128,input_shape = (3,1)))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer= 'adam', metrics=['mse'])
hist = model.fit(x,y,epochs= 10000 , batch_size= 30 ,callbacks=[es] )

#4 평가, 예측
loss = model.evaluate(x,y)

x_predict = x_predict.reshape(1,3,1)     # reshape를 하지 않으면 예측값이 3차원으로 나와서 바꿔줘야 한다.

y_predict = model.predict(x_predict)

print('loss',loss)
print('결과',y_predict)

# y_predict = 80 나오게 만들기



# Epoch 1000/1000
#  1/13 [=>............................] - ETA: 0s - loss: 0.3508WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss
# 13/13 [==============================] - 0s 2ms/step - loss: 0.4421
# 1/1 [==============================] - 0s 190ms/step - loss: 0.3486
# 1/1 [==============================] - 0s 170ms/step
# loss 0.3486323654651642
# 결과 [[77.418396]]


# Epoch 10000/10000
# 1/1 [==============================] - ETA: 0s - loss: 6.2788e-04 - mse: 6.2788e-04WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,mse
# 1/1 [==============================] - 0s 2ms/step - loss: 6.2788e-04 - mse: 6.2788e-04
# 1/1 [==============================] - 0s 195ms/step - loss: 5.9085e-04 - mse: 5.9085e-04
# 1/1 [==============================] - 0s 153ms/step
# loss [0.0005908522289246321, 0.0005908522289246321]
# 결과 [[78.10169]]


# Epoch 10000/10000
# 1/1 [==============================] - ETA: 0s - loss: 4.8942e-06 - mse: 4.8942e-06WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,mse
# 1/1 [==============================] - 0s 9ms/step - loss: 4.8942e-06 - mse: 4.8942e-06
# 1/1 [==============================] - 0s 107ms/step - loss: 4.8890e-06 - mse: 4.8890e-06
# 1/1 [==============================] - 0s 101ms/step
# loss [4.889023784926394e-06, 4.889023784926394e-06]
# 결과 [[80.734924]]


# Epoch 503: early stopping
# 1/1 [==============================] - 0s 110ms/step - loss: 1.8998e-05 - mse: 1.8998e-05
# 1/1 [==============================] - 0s 90ms/step
# loss [1.899794915516395e-05, 1.899794915516395e-05]
# 결과 [[80.191605]]