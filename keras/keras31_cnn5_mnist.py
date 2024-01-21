from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential
from keras.layers import Dense ,Conv2D , Flatten , Dropout , LeakyReLU , ELU, MaxPooling2D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import RobustScaler , StandardScaler , MaxAbsScaler , MinMaxScaler

#1 데이터
# 0.77 이상
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# print(x_train.shape,y_train.shape)                  # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape,y_test.shape)                    # (10000, 32, 32, 3) (10000, 1)
# print(np.unique(y_train,return_counts=True))        # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)

y_train= y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])


y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

es = EarlyStopping(monitor='val_loss' , mode='min' , verbose= 1 , patience = 100 ,restore_best_weights=True )

#2 모델구성
# model = Sequential()
# model.add(Conv2D(80,(3,3),input_shape = (32,32,3),activation='sigmoid'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))
# model.add(Conv2D(6,(3,3),activation='relu'))
# model.add(Conv2D(98,(3,3),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(8,activation='relu'))
# model.add(Dense(86,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(8,activation='relu'))
# model.add(Dense(104,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10,activation='softmax'))


model = Sequential()
model.add(Conv2D(150, (2, 2), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(150, (2, 2), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(150, (2, 2), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy' , optimizer='adam' , metrics=['acc'])
model.fit(x_train,y_train,epochs = 10000 ,batch_size= 1000 , verbose= 2 , callbacks=[es] , validation_split=0.2 )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1)
y_test = np.argmax(y_test,axis=1)

print('loss',loss[0])
print('loss_acc',loss[1])
print('acc', accuracy_score(y_test,y_predict) )



# loss 1.8195503950119019
# loss_acc 0.36899998784065247
# acc 0.369


# Epoch 109: early stopping
# 313/313 [==============================] - 1s 1ms/step - loss: 1.5418 - acc: 0.4487
# 313/313 [==============================] - 0s 862us/step
# loss 1.5417747497558594
# loss_acc 0.4487000107765198
# acc 0.4487


# Epoch 114: early stopping
# 313/313 [==============================] - 1s 1ms/step - loss: 1.4117 - acc: 0.5083
# 313/313 [==============================] - 0s 783us/step
# loss 1.4117342233657837
# loss_acc 0.5083000063896179
# acc 0.5083


# Epoch 314: early stopping
# 313/313 [==============================] - 1s 2ms/step - loss: 1.3455 - acc: 0.5349
# 313/313 [==============================] - 0s 882us/step
# loss 1.3454763889312744
# loss_acc 0.5349000096321106
# acc 0.5349


# Epoch 726: early stopping
# 313/313 [==============================] - 1s 1ms/step - loss: 0.9852 - acc: 0.6547
# 313/313 [==============================] - 0s 828us/step
# loss 0.9851832985877991
# loss_acc 0.654699981212616
# acc 0.6547


# Epoch 303: early stopping
# 313/313 [==============================] - 0s 1ms/step - loss: 0.9059 - acc: 0.6827
# 313/313 [==============================] - 0s 764us/step
# loss 0.9058582186698914
# loss_acc 0.682699978351593
# acc 0.6827

# Epoch 1301: early stopping
# 313/313 [==============================] - 0s 1ms/step - loss: 0.8316 - acc: 0.7098
# 313/313 [==============================] - 0s 776us/step
# loss 0.831630289554596
# loss_acc 0.7098000049591064
# acc 0.7098



