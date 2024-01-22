from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense , Conv2D , Dropout , Flatten , MaxPooling2D , BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#1

(x_train,y_train), (x_test,y_test) = cifar100.load_data()

# print(x_train.shape , y_train.shape)    # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape,y_test.shape)        # (10000, 32, 32, 3) (10000, 1)

unique , counts=np.unique(y_test, return_counts=True)   # unique를 찍을때에는 onehot을 하기전에 해야된다.
# print(unique , counts) 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train)
# print(y_test)
       # (array([0., 1.], dtype=float32), array([4950000,   50000], dtype=int64))

es =  EarlyStopping(monitor='val_loss', mode = 'min' , patience= 100 ,restore_best_weights=True , verbose=1  )


#2 모델구성
model = Sequential()
model.add(Conv2D(150,(2,2),input_shape = (32,32,3),activation='relu', strides=2 , padding='valid'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(Conv2D(14,(2,2),activation='relu',strides=2))
model.add(MaxPooling2D())
model.add(Conv2D(96,(2,2),activation='relu',padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(9,(2,2),activation='relu' , padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(91,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7,activation='relu'))
model.add(Dense(100,activation='softmax'))

#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam' , metrics= ['acc'] )
model.fit(x_train,y_train, epochs= 100000 , batch_size= 1000 , callbacks=[es] , verbose= 1 , validation_split= 0.2  )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print('loss =',loss)

y_test = np.argmax(y_test,axis=1)
y_predict = np.argmax(y_predict,axis=1)

def ACC(aaa,bbb):
    return accuracy_score(aaa,bbb)
acc = ACC(y_test,y_predict)

print('ACC = ' , acc)


plt.imshow(x_train[15],'gray')
plt.show()




# Epoch 579: early stopping
# 313/313 [==============================] - 0s 1ms/step - loss: 2.9219 - acc: 0.2607
# 313/313 [==============================] - 0s 800us/step
# loss = [2.9218697547912598, 0.260699987411499]
# ACC =  0.2607


# Epoch 930: early stopping
# 313/313 [==============================] - 1s 1ms/step - loss: 2.7536 - acc: 0.2864
# 313/313 [==============================] - 0s 809us/step
# loss = [2.7536168098449707, 0.2863999903202057]
# ACC =  0.2864


# Epoch 843: early stopping
# 313/313 [==============================] - 1s 1ms/step - loss: 2.7895 - acc: 0.2732
# 313/313 [==============================] - 0s 809us/step
# loss = [2.7894644737243652, 0.27320000529289246]
# ACC =  0.2732


# Epoch 205: early stopping
# 313/313 [==============================] - 0s 1ms/step - loss: 0.9167 - acc: 0.6942
# 313/313 [==============================] - 0s 695us/step
# loss 0.9166645407676697
# loss_acc 0.6941999793052673
# acc 0.6942
