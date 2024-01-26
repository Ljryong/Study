# 과제

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense , Conv2D ,Flatten , MaxPooling2D , Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import time
import datetime

############################ 사진이 꺠져있음 , 확인방법 #########
# 1. 사진의 파일 크기를 확인하여 제거할 수 있다.


#1 데이터

np_path = 'c:/_data/_save_npy//'
x = np.load(np_path + 'keras39_catdog_x_train.npy')
y = np.load(np_path + 'keras39_catdog_y_train.npy')
test = np.load(np_path + 'keras39_catdog_test.npy')

x_train , x_test , y_train , y_test = train_test_split(x,y,random_state= 51 , stratify= y  , shuffle=True , test_size=0.3 )

es = EarlyStopping(monitor='val_loss' , mode = 'min' , patience= 10 , restore_best_weights=True , verbose= 1  )



#2 모델구성
model = Sequential()
model.add(Conv2D(64,(2,2),input_shape = (100,100,3) , strides=1 , activation='relu' ))
model.add(MaxPooling2D())
model.add(Conv2D(32,(2,2), activation='relu' ))
model.add(Conv2D(28,(2,2), activation='relu' ))
model.add(Conv2D(24,(2,2), activation='relu' ))
model.add(Conv2D(20,(2,2), activation='relu' ))
model.add(Conv2D(16,(2,2), activation='relu' ))
model.add(Conv2D(12,(2,2), activation= 'relu' ))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(80,activation='relu'))
model.add(Dense(40, activation= 'relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# model.summary()

#3 컴파일, 훈련
model.compile(loss= 'binary_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train, epochs = 1 , batch_size= 10 , validation_split= 0.2, verbose= 1 ,callbacks=[es])


#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(test)

print('loss',loss)

y_test = np.round(y_test)

y_predict = np.round(y_predict.reshape(-1))
print(y_predict.shape)


import os

folder_path = 'C:\\_data\\image\\catdog\\Test\\test'
file_list = os.listdir(folder_path)
file_names = np.array([os.path.splitext(file_name)[0] for file_name in file_list])

y_submit = pd.DataFrame({'id' : file_names, 'Target' : y_predict })

print(y_submit['Target'])
csv_path = 'C:\\_data\\kaggle\\catdog\\'


date = datetime.datetime.now().strftime("%m%d_%H%M")    #01171053   

y_submit.to_csv(csv_path + date  + ".csv", index=False)

print('loss = ' , loss)

file_acc = str(round(loss[1], 6))
date = datetime.datetime.now().strftime("%m%d_%H%M")

model.save('C:\\_data\\_save\\models\\kaggle\\cat_dog\\'+ date + '_' + file_acc +'_cnn.hdf5')

    
# time :  128.8634068965912  (150,150)

# time :  82.54952311515808  (100,100)

# Epoch 10/10       (10,10)
# 1120/1120 [==============================] - 2s 2ms/step - loss: 0.6480 - acc: 0.6103 - val_loss: 0.6590 - val_acc: 0.5979
# 188/188 [==============================] - 0s 2ms/step - loss: 0.6548 - acc: 0.6062
# 188/188 [==============================] - 0s 621us/step
# loss [0.6548140645027161, 0.606166660785675]
# Acc =  0.6061666666666666
# time :  44.61295771598816



# Epoch 10/10       (80,80)
# 1120/1120 [==============================] - 7s 6ms/step - loss: 0.5977 - acc: 0.6718 - val_loss: 0.6838 - val_acc: 0.5879
# 188/188 [==============================] - 1s 4ms/step - loss: 0.6665 - acc: 0.6025
# 188/188 [==============================] - 1s 3ms/step
# loss [0.6664801239967346, 0.6025000214576721]
# Acc =  0.6025
# time :  69.61687755584717




# Epoch 83: early stopping
# 188/188 [==============================] - 1s 5ms/step - loss: 0.6887 - acc: 0.5430
# 188/188 [==============================] - 1s 5ms/step
# loss [0.6886634230613708, 0.5429999828338623]
# Acc =  0.543
# time :  75.94417262077332

# Epoch 38: early stopping
# 188/188 [==============================] - 1s 5ms/step - loss: 0.6522 - acc: 0.6113
# 188/188 [==============================] - 1s 4ms/step
# loss [0.6521573662757874, 0.6113333106040955]
# Acc =  0.6113333333333333
# time :  76.63124442100525


# loss [0.6932511925697327, 0.5005263090133667]
# Acc =  0.5005263157894737
# time :  23.81118869781494

# Epoch 95/100
#  1/17 [>.............................] - ETA: 0s - loss: 0.0020 - acc: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 1700 batches). You may need to use the repeat() function when building your dataset.
# WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,acc
# 17/17 [==============================] - 0s 16ms/step - loss: 0.0011 - acc: 1.0000
# 5/5 [==============================] - 0s 4ms/step - loss: 1.3497e-04 - acc: 1.0000
# 5/5 [==============================] - 0s 7ms/step
# loss [0.0001349706872133538, 1.0]
# Acc =  1.0