import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


#1 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))
# print(x)

# x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.35, shuffle=False)
# print(x_train)
# x_val , x_test , y_val , y_test = train_test_split(x_test,y_test,test_size = 0.5 , shuffle= False)
# print(x_test)

# 둘 다 되는데 밑에가 좀 더 좋게 나옴

x_train = x[:10]
print(x_train)
x_val = x[10 : 13]
print(x_val)
x_test = x[13:]
print(x_test)
y_train = y[:10]
y_val = y[10:13]
y_test = y[13:]

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
# 10/10 [==============================] - 0s 2ms/step - loss: 8.8711e-13 - val_loss: 0.0000e+00
# 1/1 [==============================] - 0s 48ms/step - loss: 6.0633e-13
# 1/1 [==============================] - 0s 52ms/step
# 6.063298192519884e-13
# [[14.      ]
#  [14.999999]
#  [15.999999]]
# batch = 1