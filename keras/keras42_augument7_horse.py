import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential , Model
from keras.layers import Dense , Conv2D , MaxPooling2D , Flatten , Dropout , Input
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

path_train = 'C:\_data\image\horse_human\\'


xy = train_datagen.flow_from_directory(
    path_train,
    target_size=(100,100),
    color_mode='rgb',
    class_mode='binary',
    batch_size=1000,
    shuffle=True
    
)

x=[]
y=[]
for i in range(len(xy)):
    a, b = xy.next()
    x.append(a)
    y.append(b)

x = np.concatenate(x,axis=0)
y = np.concatenate(y,axis=0)

# 증폭 시점 고쳐보기

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3, random_state= 0 ,  stratify=y , shuffle=True )

print('split ok')

augument_size = 100

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
randidx = np.random.randint(x_train.shape[0], size=augument_size)       # x_train.shape[0] : 수량체크
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
input = Input(shape=(100,100,3))
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


# 증폭
# Epoch 13: early stopping
# 10/10 [==============================] - 0s 6ms/step - loss: 0.0864 - acc: 0.9676
# 10/10 [==============================] - 0s 3ms/step
# loss 0.08642079681158066
# acc 0.9676375389099121
# ACC 0.9676375404530745