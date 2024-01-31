from keras.models import Sequential , Model
from keras.layers import Dense , Input , Dropout , Conv2D , MaxPooling2D ,Flatten , LSTM , Conv1D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

#1 데이터 

np_path = 'c:/_data/_save_npy//'

x = np.load(np_path + 'keras39_07_x.npy')
y = np.load(np_path + 'keras39_07_y.npy')

x = x.reshape(-1,900,300)

x_train, x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state= 53 ,stratify=y , shuffle=True  )


es = EarlyStopping(monitor='val_loss' , mode = 'min' , restore_best_weights=True , verbose= 1 , patience= 10  )

#2 모델구성
input = Input(shape=(900,300))
c1 = Conv1D(4,3,activation='relu')(input)
c2 = Conv1D(41,2,activation='relu')(c1)
c3 = Conv1D(4,3,activation='relu')(c2)
f1 = Flatten()(c3)
d1 = Dense(1,activation='relu')(f1)
drop1 = Dropout(0.2)(d1)
d2 = Dense(2,activation='relu')(drop1)
drop2 = Dropout(0.2)(d1)
d3 = Dense(1,activation='relu')(drop2)
drop3 = Dropout(0.2)(d1)
output = Dense(2,activation='softmax')(drop3)

model = Model(inputs = input , outputs = output )

model.summary()



#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train,epochs= 10000 , validation_split=0.2 , verbose= 1 , callbacks=[es] , batch_size=1000 )

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


# LSTM
# Epoch 39: early stopping
# 10/10 [==============================] - 1s 114ms/step - loss: 142.9508 - acc: 0.4725
# 10/10 [==============================] - 1s 108ms/step
# loss 142.9507598876953
# acc 0.47249191999435425
# ACC 0.47249190938511326


# Conv1D
# Epoch 34: early stopping
# 10/10 [==============================] - 0s 11ms/step - loss: 0.6083 - acc: 0.7346
# 10/10 [==============================] - 0s 10ms/step
# loss 0.6083233952522278
# acc 0.7346278429031372
# ACC 0.7346278317152104











