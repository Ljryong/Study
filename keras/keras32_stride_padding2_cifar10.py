from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense , Conv2D , Dropout , Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical

#1 데이터
(x_train,y_train) , (x_test,y_test) = cifar10.load_data()

print(x_train.shape,y_train.shape)      # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape,y_test.shape)        # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train))


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)
print(y_test)

es = EarlyStopping(monitor= 'val_loss' , mode = 'min' , patience = 100 , restore_best_weights=True , verbose= 1 )

#2 모델구성 
model = Sequential()
model.add(Conv2D(50,(2,2),input_shape = (32,32,3), padding='same' , strides=1))
model.add(Dropout(0.2))
model.add(Conv2D(35,(2,2),padding='same'))
model.add(Conv2D(48,(2,2),padding='same'))
model.add(Flatten())
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(27))
model.add(Dense(10,activation='softmax'))

model.summary()

#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train,epochs = 100000 ,batch_size= 1000 , verbose= 2 , callbacks=[es] , validation_split=0.2 )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1)
y_test = np.argmax(y_test,axis=1)

print('loss',loss[0])
print('loss_acc',loss[1])
print('acc', accuracy_score(y_test,y_predict) )