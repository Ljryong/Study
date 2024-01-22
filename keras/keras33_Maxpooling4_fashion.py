from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense , Conv2D , Dropout , Flatten , MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1 데이터 
(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()

print(x_train.shape,y_train.shape)      # (60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)        # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

print(np.unique(y_test))       # [0 1 2 3 4 5 6 7 8 9]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

es = EarlyStopping(monitor='val_loss' , mode='min' , patience=10 , restore_best_weights=True , verbose= 1 )

#2 모델구성
model = Sequential(Conv2D(41, (2,2) , input_shape = (28,28,1) , activation='relu' , padding='same' , strides=1 ))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(Conv2D(21 , (2,2) , activation='relu', padding='same' , strides=1))
model.add(Dropout(0.2))
model.add(Conv2D(38 , (2,2) , activation='relu', padding='same' , strides=1))
model.add(Dropout(0.3))
model.add(Conv2D(19 , (2,2) , activation='relu', padding='same' , strides=1))
model.add(Dropout(0.2))
model.add(Conv2D(48 , (2,2) , activation='relu', padding='same' , strides=1))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(27,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(9,activation='relu'))
model.add(Dense(10,activation='softmax'))



#3 컴파일, 훈련
model.compile(loss='categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train,epochs= 10000 , batch_size = 1000 , verbose= 1 , validation_split= 0.2 , callbacks=[es] )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print('loss = ' , loss)

y_test = np.argmax(y_test,axis=1)
y_predict = np.argmax(y_predict,axis=1)

print('acc = ',accuracy_score(y_test,y_predict))

plt.imshow(x_train[1] , 'gray')         # 'gray' = 사진의 색깔을 바꾸는 행동
plt.show()


# Epoch 116: early stopping
# 313/313 [==============================] - 1s 1ms/step - loss: 0.2274 - acc: 0.9211
# 313/313 [==============================] - 0s 854us/step
# loss =  [0.22739548981189728, 0.9211000204086304]
# acc =  0.9211


# Epoch 501: early stopping
# 313/313 [==============================] - 1s 3ms/step - loss: 0.2092 - acc: 0.9295
# 313/313 [==============================] - 1s 2ms/step
# loss =  [0.20921477675437927, 0.9294999837875366]
# acc =  0.9295

# Epoch 124: early stopping
# 313/313 [==============================] - 1s 1ms/step - loss: 0.2452 - acc: 0.9149
# 313/313 [==============================] - 0s 802us/step
# loss =  [0.2452477514743805, 0.914900004863739]
# acc =  0.9149


















