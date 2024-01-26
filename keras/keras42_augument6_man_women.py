import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense , Conv2D , MaxPooling2D , Flatten , Dropout
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

#1 데이터

train_datagen = ImageDataGenerator(1./255,)
                                #    horizontal_flip=True,
                                #    vertical_flip=True,
                                #    zoom_range=0.2,
                                #    shear_range=0.2,
                                #    rotation_range=50,
                                #    width_shift_range=0.2,
                                #    height_shift_range=0.2)

test_datagen = ImageDataGenerator(1./255)

path_train = 'C:/_data/kaggle/man_women/train//'
path_test = 'C:/_data/kaggle/man_women/submit//'


xy_train = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),
    color_mode='rgb',
    class_mode='binary',
    batch_size=1000,
    shuffle=True
    
)

submit = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    color_mode='rgb',
    class_mode='binary',
    batch_size=1000,
    shuffle=False

)

x=[]
y=[]
for i in range(len(xy_train)):
    a, b = xy_train.next()
    x.append(a)
    y.append(b)

x = np.concatenate(x,axis=0)
y = np.concatenate(y,axis=0)

# 증폭 시점 고쳐보기

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3, random_state= 0 ,  stratify=y , shuffle=True )

print('split ok')

augument_size = 10000

data_gen = ImageDataGenerator(
    # rescale=1/255. ,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    fill_mode='nearest',
    shear_range=10
)
randidx = np.random.randint(x_train.shape[0], size=augument_size)
x_aug = x_train[randidx].copy()
y_aug = y_train[randidx].copy()
x_aug = data_gen.flow(
    x_aug, y_aug,
    batch_size=augument_size,
    shuffle=False
).next()[0]
x_train = np.concatenate((x_train, x_aug), axis=0)
y_train = np.concatenate((y_train, y_aug), axis=0)

es =  EarlyStopping(monitor='val_loss', mode = 'min' , patience= 10 ,restore_best_weights=True , verbose=1  )



#모델구성
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
model = Sequential()

model.add(Conv2D(64, (2,2), input_shape=(100,100,3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode = 'min', patience= 30, restore_best_weights=True)

#컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs= 1000, batch_size= 50, validation_split= 0.2, callbacks=[es])

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(submit)

print('loss : ', loss[0])
print('acc : ', loss[1])

y_test = np.round(y_test)
y_predict = np.round(y_predict)

# print('ACC : ' , accuracy_score(y_test,y_predict))

print(y_predict)





# Epoch 46/1000
# 150/150 [==============================] - 2s 14ms/step - loss: 0.0171 - acc: 0.4259 - val_loss: 0.0158 - val_acc: 0.4383
# 125/125 [==============================] - 1s 4ms/step - loss: 0.0159 - acc: 0.4282
# 1/1 [==============================] - 0s 89ms/step
# loss :  0.01591419242322445
# acc :  0.42824944853782654
# [[0.]]



# 1/1 [==============================] - 0s 66ms/step
# loss :  0.6654999256134033
# acc :  0.6223564743995667
# [[1.]]