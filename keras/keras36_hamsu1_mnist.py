import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential , Model
from keras.layers import Dense , Conv2D , Flatten  , MaxPooling2D  , Input     # Flatten : 평평한
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

#1 데이터
(x_train , y_train), (x_test, y_test)  =  mnist.load_data()
print(x_train.shape , y_train.shape)    # (60000, 28, 28) (60000,)
print(x_test.shape , y_test.shape)      # (10000, 28, 28) (10000,)

print(x_train)
print(x_train[0])
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))
print(pd.value_counts(y_test))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)
# print(x_train.shape[0])         # 60000

# x_test = x_test.reshape(x_test[0],x_test[1],x_test[2],1)        # 위에 10000,28,28,1 이랑 같다. 이렇게 쓰는게 나중에 전처리하고 test가 달라졌을 때 좋을수도 있다.

print(x_train.shape , x_test.shape)    # (60000, 784) (10000, 784)

es = EarlyStopping(monitor='val_loss'  , mode='min' , patience=50 ,restore_best_weights=True , verbose= 2   )

#2 모델구성 DNN형
# model = Sequential()
# model.add(Dense(600, input_shape = (784,), activation='relu'))
# model.add(Dense (650 , activation='relu'))
# model.add(Dense (450 , activation='relu'))
# model.add(Dense (550 , activation='relu'))
# model.add(Dense (350 , activation='relu'))
# model.add(Dense (450 , activation='relu'))
# model.add(Dense (250 , activation='relu'))
# model.add(Dense (250 , activation='relu'))
# model.add(Dense (100 , activation='relu'))
# model.add(Dense (70 , activation='relu'))
# model.add(Dense (10 , activation='softmax'))

#2-1 모델구성 함수형
input = Input(shape=(784,))
d1 = Dense(600,activation='relu')(input)
d2 = Dense(650,activation='relu')(d1)
d3 = Dense(450,activation='relu')(d2)
d4= Dense(550,activation='relu')(d3)
d5 = Dense(350,activation='relu')(d4)
d6 = Dense(450,activation='relu')(d5)
d7 = Dense(250,activation='relu')(d6)
d8 = Dense(250,activation='relu')(d7)
d9 = Dense(100,activation='relu')(d8)
d10 = Dense(70,activation='relu')(d9)
output = Dense(10,activation='softmax')(d10)
model = Model(inputs = input , outputs = output)


#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train,epochs=10, batch_size=1000 , validation_split=0.2 , verbose= 1 , callbacks=[es])


#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print('loss =' , loss)
y_test = np.argmax(y_test,axis=1)
y_predict = np.argmax(y_predict,axis=1)

print(accuracy_score(y_test,y_predict))

# Epoch 55: early stopping
# 313/313 [==============================] - 0s 1ms/step - loss: 0.1036 - acc: 0.9684
# 313/313 [==============================] - 0s 762us/step
# loss = [0.10358822345733643, 0.9684000015258789]
# 0.9684


# 함수
# loss = [0.11261005699634552, 0.9735000133514404]
# 0.9735


