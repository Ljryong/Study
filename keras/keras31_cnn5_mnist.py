from keras.datasets import cifar10
import numpy as np
from keras.models import Sequential
from keras.layers import Dense ,Conv2D , Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

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
model = Sequential()
model.add(Conv2D(30,(3,3),input_shape = (32,32,3),activation='relu'))
model.add(Conv2D(6,(2,2),activation='relu'))
model.add(Conv2D(84,(3,3),activation='relu'))
model.add(Conv2D(8,(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(200,activation='relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(86,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(104,activation='relu'))
model.add(Dense(10,activation='softmax'))

#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy' , optimizer='adam' , metrics=['acc'])
model.fit(x_train,y_train,epochs = 1000000 ,batch_size= 1000 , verbose= 2 , callbacks=[es] , validation_split=0.2 )

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




