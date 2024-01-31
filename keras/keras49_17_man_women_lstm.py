from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# datapath = 'C:/Workspace/AIKONG/_data/'
datapath = 'C:/_data/'
np_path = datapath + '_save_npy/'

x = np.load(np_path + 'keras39_5_x_train.npy')
y = np.load(np_path + 'keras39_5_y_train.npy')
test = np.load(np_path + 'keras39_5_test.npy')

x = x.reshape(-1,300,100)
test = test.reshape(-1,300,100)

print(x.shape)
print(y.shape)
print(test.shape)

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.3 , random_state= 0 , stratify=y ,shuffle= True)


# x_train = x_train/255.
# x_test = x_test/255.




#모델구성
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D , LSTM
model = Sequential()

model.add(LSTM(30, input_shape=(300,100), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode = 'min', patience= 30, restore_best_weights=True)

#컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs= 1000, batch_size= 1000, validation_split= 0.2, callbacks=[es])

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(test)

print('loss : ', loss[0])
print('acc : ', loss[1])

y_test = np.round(y_test)
y_predict = np.round(y_predict)

# print('ACC : ' , accuracy_score(y_test,y_predict))

print(y_predict)




# LSTM
# Epoch 448/1000
# 2/2 [==============================] - 1s 298ms/step - loss: nan - acc: 0.4239 - val_loss: nan - val_acc: 0.4332
# 32/32 [==============================] - 1s 38ms/step - loss: 0.6659 - acc: 0.6183
# 1/1 [==============================] - 0s 120ms/step
# loss :  0.665940523147583
# acc :  0.6183282732963562
# [[0.]]