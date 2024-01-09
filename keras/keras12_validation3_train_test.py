import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


#1 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
# print(x)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.35, shuffle = False )
print(x_train)
x_val , x_test , y_val , y_test = train_test_split(x_test,y_test,test_size = 0.5 , shuffle= False)
print(x_test)

#2 모델구성
model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

# #3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs = 500 , batch_size = 1 , validation_data=(x_val,y_val))

# #4 평가,예측
loss = model.evaluate(x_test,y_test)
result = model.predict([14,15,16])
print(loss)
print(result)


# Epoch 500/500
# 10/10 [==============================] - 0s 2ms/step - loss: 1.5348e-13 - val_loss: 1.8190e-12
# 1/1 [==============================] - 0s 50ms/step - loss: 7.8823e-12
# 1/1 [==============================] - 0s 63ms/step
# 7.882287704485957e-12
# [[13.999999]
#  [14.999997]
#  [15.999996]]
# batch = 1