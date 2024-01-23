from keras.datasets import fashion_mnist
from keras.models import Sequential , Model
from keras.layers import Dense , Conv2D , Dropout , Flatten , MaxPooling2D , Input
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler , StandardScaler


#1 데이터 
(x_train,y_train), (x_test,y_test) = fashion_mnist.load_data()

print(x_train.shape,y_train.shape)         # (60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)           # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

print(np.unique(y_test))

# x_train = x_train/255
# x_test = x_test/255

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

es = EarlyStopping(monitor='val_loss' , mode = 'min' , patience= 100  , restore_best_weights=True , verbose= 1 )  

#2 모델구성
# model = Sequential()
# model.add(Dense(1000,input_shape = (784,)))
# model.add(Dense(750,activation='relu'))
# model.add(Dense(500,activation='relu'))
# model.add(Dense(600,activation='relu'))
# model.add(Dense(350,activation='relu'))
# model.add(Dense(400,activation='relu'))
# model.add(Dense(150,activation='relu'))
# model.add(Dense(200,activation='relu'))
# model.add(Dense(50,activation='relu'))
# model.add(Dense(10,activation='softmax'))

#2-1
input = Input(shape=(784,))
d1 = Dense(1000)(input)
d2 = Dense(750,activation='relu')(d1)
d3 = Dense(500,activation='relu')(d2)
d4 = Dense(600,activation='relu')(d3)
d5 = Dense(350,activation='relu')(d4)
d6 = Dense(400,activation='relu')(d5)
d7 = Dense(150,activation='relu')(d6)
d8 = Dense(200,activation='relu')(d7)
d9 = Dense(50,activation='relu')(d8)
output = Dense(10,activation='relu')(d9)
model = Model(inputs = input , outputs = output)


#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train, epochs= 10 , batch_size= 1000 , validation_split= 0.2 , verbose= 1 , callbacks=[es] )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print('loss = ' , loss)

y_test = np.argmax(y_test,axis=1)
y_predict = np.argmax(y_predict,axis=1)

print(accuracy_score(y_test,y_predict))




# Epoch 117: early stopping
# 313/313 [==============================] - 0s 1ms/step - loss: 0.3388 - acc: 0.8808
# 313/313 [==============================] - 0s 674us/step
# loss =  [0.33875906467437744, 0.8808000087738037]
# 0.8808



# 함수
# loss =  [5.886749744415283, 0.24549999833106995]
# 0.2455


# minmaxscaler
# loss =  [2.9420430660247803, 0.2906000018119812]
# 0.2906

# minmaxscaler2
# loss =  [2.6418793201446533, 0.4765999913215637]
# 0.4766

# -1 ~ 1
# loss =  [2.8442726135253906, 0.3474000096321106]
# 0.3474

# StandardScaler
# loss =  [2.8442726135253906, 0.3474000096321106]
# 0.3474