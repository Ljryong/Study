from keras.models import Sequential , Model
from keras.layers import Dense , Input , Dropout , Conv2D , MaxPooling2D ,Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

#1 데이터 

np_path = 'c:/_data/_save_npy//'

x = np.load(np_path + 'keras39_11_x.npy')
y = np.load(np_path + 'keras39_11_y.npy')


x_train, x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state= 53 ,stratify=y , shuffle=True  )


es = EarlyStopping(monitor='val_loss' , mode = 'min' , restore_best_weights=True , verbose= 1 , patience= 10  )

#2 모델구성
input = Input(shape=(300,300,3))
c1 = Conv2D(64,(3,3),activation='relu')(input)
max1 = MaxPooling2D()(c1)
c2 = Conv2D(128,(2,2),activation='relu')(max1)
max2 = MaxPooling2D()(c2)
# c3 = Conv2D(64,(2,2),activation='relu')(max2)
# max3 = MaxPooling2D()(c3)
# c4 = Conv2D(128,(2,2),activation='relu' , strides=2 )(max3)
flat = Flatten()(max2)
d1 = Dense(64,activation='relu')(flat)
drop1 = Dropout(0.2)(d1)
d2 = Dense(128,activation='relu')(drop1)
drop2 = Dropout(0.2)(d1)
d3 = Dense(64,activation='relu')(drop2)
drop3 = Dropout(0.2)(d1)
output = Dense(1,activation='sigmoid')(drop3)

model = Model(inputs = input , outputs = output )

model.summary()

#3 컴파일, 훈련
model.compile(loss = 'binary_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train,epochs= 10000 , validation_split=0.2 , verbose= 1 , callbacks=[es] , batch_size=32 )

#4 평가, 예측
loss = model.evaluate(x_test ,y_test)
y_predict = model.predict(x_test)

print('loss' , loss[0])
print('acc' , loss[1])

y_test = np.round(y_test)
y_predict = np.round(y_predict)


print('ACC',accuracy_score(y_test,y_predict))

# Epoch 23: early stopping
# 10/10 [==============================] - 0s 36ms/step - loss: 0.0186 - acc: 0.9903
# 10/10 [==============================] - 0s 22ms/step
# loss 0.018589725717902184
# acc 0.9902912378311157
# ACC 0.9902912621359223

# Epoch 37: early stopping
# 10/10 [==============================] - 0s 38ms/step - loss: 0.0032 - acc: 0.9968
# 10/10 [==============================] - 0s 23ms/step
# loss 0.003229274181649089
# acc 0.9967637658119202
# ACC 0.9967637540453075


# Epoch 39: early stopping
# 10/10 [==============================] - 0s 31ms/step - loss: 0.0023 - acc: 1.0000
# 10/10 [==============================] - 0s 21ms/step
# loss 0.002332329051569104
# acc 1.0
# ACC 1.0


