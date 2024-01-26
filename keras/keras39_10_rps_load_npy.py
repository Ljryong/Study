from keras.models import Sequential ,Model
from keras.layers import Dense , Input , Dropout , MaxPooling2D , Conv2D ,Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


#1 데이터

np_path = 'c:/_data/_save_npy//'

x = np.load(np_path + 'keras39_09_x.npy')
y = np.load(np_path + 'keras39_09_y.npy')


x_train , x_test, y_train ,y_test = train_test_split(x,y, test_size=0.3 , random_state= 0 , shuffle=True, stratify=y )

es = EarlyStopping(monitor='val_loss' , mode='min' , patience= 10 , restore_best_weights=True , verbose=1 )

#2 모델구성
input = Input(shape=(150,150,3))
c1 = Conv2D(64,(3,3),activation='relu')(input)
max1 = MaxPooling2D()(c1)
c2 = Conv2D(128,(3,3),activation='relu')(max1)
max2 = MaxPooling2D()(c2)
# c3 = Conv2D(64,(3,3),activation='relu')(max2)
# max3 = MaxPooling2D()(c3)
# c4 = Conv2D(128,(2,2),activation='relu' , strides=2 )(max3)
flat = Flatten()(max2)
d1 = Dense(64,activation='relu')(flat)
drop1 = Dropout(0.3)(d1)
d2 = Dense(128,activation='relu')(drop1)
drop2 = Dropout(0.3)(d1)
d3 = Dense(64,activation='relu')(drop2)
drop3 = Dropout(0.3)(d1)
output = Dense(3,activation='softmax')(drop3)

model = Model(inputs = input , outputs = output )

#3 컴파일, 훈련
model.compile(loss ='categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train, epochs=10000, batch_size=20 , validation_split= 0.2 , verbose= 1 , callbacks=[es] )

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