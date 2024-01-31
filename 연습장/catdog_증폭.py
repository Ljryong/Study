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

path_train = 'C:\_data\image\catdog\Train\\'
path_test = 'C:\_data\image\catdog\Test\\'


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


#2 모델구성
model = Sequential()
model.add(Conv2D(94,(3,3),input_shape = (100,100,3), padding='valid' , strides=1 , activation='relu' ))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(12,(2,2), activation='relu' ))

model.add(Conv2D(81,(2,2), activation='relu' ))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(9,(3,3), activation='relu' ))

model.add(Conv2D(92,(2,2), activation='relu' ))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(11,(2,2),activation='relu'))

model.add(Flatten())

model.add(Dense(73,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(12))

model.add(Dense(42,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))


#3 컴파일, 훈련
model.compile(loss= 'binary_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train, epochs = 100 , batch_size= 10 , validation_split= 0.2, verbose= 1 ,callbacks=[es])


#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print('loss',loss)

y_test = np.round(y_test)
y_predict = np.round(y_predict)

print('Acc = ' , accuracy_score(y_test,y_predict))




# Epoch 17: early stopping
# 282/282 [==============================] - 1s 4ms/step - loss: 0.6932 - acc: 0.5003
# 282/282 [==============================] - 1s 4ms/step
# loss [0.6931765675544739, 0.500333309173584]
# Acc =  0.5003333333333333