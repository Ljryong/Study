from keras.models import Sequential ,Model
from keras.layers import Dense , Input , Dropout , MaxPooling2D , Conv2D ,Flatten , LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


#1 데이터

np_path = 'c:/_data/_save_npy//'

x = np.load(np_path + 'keras39_09_x.npy')
y = np.load(np_path + 'keras39_09_y.npy')

x = x.reshape(-1,450,150)

x_train , x_test, y_train ,y_test = train_test_split(x,y, test_size=0.3 , random_state= 0 , shuffle=True, stratify=y )

es = EarlyStopping(monitor='val_loss' , mode='min' , patience= 10 , restore_best_weights=True , verbose=1 )

#2 모델구성
input = Input(shape=(450,150))
l1 = LSTM(64,activation='relu')(input)
d1 = Dense(64,activation='relu')(l1)
drop1 = Dropout(0.3)(d1)
d2 = Dense(128,activation='relu')(drop1)
drop2 = Dropout(0.3)(d1)
d3 = Dense(64,activation='relu')(drop2)
drop3 = Dropout(0.3)(d1)
output = Dense(3,activation='softmax')(drop3)

model = Model(inputs = input , outputs = output )

#3 컴파일, 훈련
model.compile(loss ='categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train, epochs=10000, batch_size=10000 , validation_split= 0.2 , verbose= 1 , callbacks=[es] )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print('loss' , loss[0])
print('acc' , loss[1])

y_test = np.round(y_test)
y_predict = np.round(y_predict)

print('Acc =' ,accuracy_score(y_test, y_predict))

# Epoch 245: early stopping
# 24/24 [==============================] - 0s 7ms/step - loss: 0.0000e+00 - acc: 1.0000
# 24/24 [==============================] - 0s 7ms/step
# loss 0.0
# acc 1.0
# Acc = 1.0

# LSTM
# Epoch 13: early stopping
# 24/24 [==============================] - 3s 137ms/step - loss: 5.3536 - acc: 0.3399
# 24/24 [==============================] - 3s 132ms/step
# loss 5.353571891784668
# acc 0.3399471044540405