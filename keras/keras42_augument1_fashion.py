from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential , Model
from keras.layers import Dense , Input , Conv2D , MaxPooling2D , Dropout ,Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical

(x_train,y_train) , (x_test,y_test) = fashion_mnist.load_data()

x_train= x_train/255.
x_test= x_test/255.


train_datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip=True,           # 수평 뒤집기
    vertical_flip=True,             # 수직 뒤집기
    width_shift_range=0.2,          # 가로이동 비율
    height_shift_range=0.2,         # 세로이동 비율
    rotation_range=50,              # 회전 각도 조절
    zoom_range=0.2,                 # 축소, 확대 비율 조절
    shear_range=0.8,                # 찌끄러뜨리거나 , 누르기 
    fill_mode='nearest',
    
)

argumet_size = 40000

randidx = np.random.randint(x_train.shape[0], size = argumet_size)               # 랜덤한 인트값을 뽑는다.
        # np.random.randint(60000,40000)                                        # 60000개 중 40000개를 랜덤으로 뽑는다

print(randidx)      # [ 4709  5920 14810 ... 45827 18883  1793]
print(np.min(randidx), np.max(randidx) )                # 0 59999

x_augummented = x_train[randidx].copy()          # 원데이터에 영향을 미치지 않기 위해서 .copy() 를 쓴다

y_augummented = y_train[randidx].copy()

print(x_augummented)
print(x_augummented.shape)          # (40000, 28, 28)
print(y_augummented)
print(y_augummented.shape)          # (40000,)


x_augummented = x_augummented.reshape(40000,28,28,1)
            # = x_augummented.reshape(-1,28,28,1)
            # = x_augummented.reshape(x_augummented[0],x_augummented[1],x_augummented[2],1)
            
print(x_augummented)       
print(x_augummented.shape)        # (40000, 28, 28, 1)
            
            
x_augummented = train_datagen.flow(
    x_augummented, y_augummented ,
    batch_size=argumet_size,
    shuffle = False
).next()[0]                    # .next 뒤에 [0] 을 쓰면 x 값만 나온다.


print(x_augummented[0])         # [0]은 x [1]은 y

print(x_train.shape)                # (60000, 28, 28)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)


print(x_train.shape,x_augummented.shape)


x_train = np.concatenate((x_train,x_augummented))              #concatenate: 사슬처럼 엮다
y_train = np.concatenate((y_train,y_augummented))
print(x_train.shape,y_train.shape)          # (100000, 28, 28, 1) (100000,)


es = EarlyStopping(monitor= 'val_loss' , mode = 'min' , patience= 10 , restore_best_weights=True , verbose=1    ) 


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)




#2 모델구성
model = Sequential(Conv2D(41, (2,2) , input_shape = (28,28,1) , activation='relu'  , strides=1 ))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(Conv2D(21 , (2,2) , activation='relu', strides=1))
model.add(Dropout(0.2))
model.add(Conv2D(38 , (2,2) , activation='relu', strides=1))
model.add(Dropout(0.3))
model.add(Conv2D(19 , (2,2) , activation='relu', strides=1))
model.add(Dropout(0.2))
model.add(Conv2D(48 , (2,2) , activation='relu' , strides=1))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(27,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(9,activation='relu'))
model.add(Dense(10,activation='softmax'))

#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train, epochs= 10 , batch_size= 1000 , validation_split= 0.2 , verbose= 1 , callbacks=[es] )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print('loss = ' , loss)

y_test = np.argmax(y_test,axis=1)
y_predict = np.argmax(y_predict,axis=1)

print(accuracy_score(y_test,y_predict))




# Epoch 117: early stopping
# 313/313 [==============================] - 0s 1ms/step - loss: 0.3388 - acc: 0.8808
# 313/313 [==============================] - 0s 674us/step
# loss =  [0.33875906467437744, 0.8808000087738037]
# 0.8808



# 함수
# loss =  [5.886749744415283, 0.24549999833106995]
# 0.2455


# 증폭 
# Epoch 10/10
# 80/80 [==============================] - 1s 12ms/step - loss: 0.7851 - acc: 0.7131 - val_loss: 1.4243 - val_acc: 0.4734
# 313/313 [==============================] - 0s 1ms/step - loss: 0.3937 - acc: 0.8576
# 313/313 [==============================] - 0s 686us/step
# loss =  [0.39366403222084045, 0.8575999736785889]
# 0.8576






