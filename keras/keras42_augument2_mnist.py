from keras.datasets import mnist
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

(x_train,y_train) , (x_test,y_test) = mnist.load_data()

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

argumet_size = 10000

randidx = np.random.randint(x_train.shape[0], size = argumet_size) 
       
print(randidx)     
print(np.min(randidx), np.max(randidx) )

x_augummented = x_train[randidx].copy() 

y_augummented = y_train[randidx].copy()

print(x_augummented)
print(x_augummented.shape)         
print(y_augummented)
print(y_augummented.shape)         


x_augummented = x_augummented.reshape(10000,28,28,1)
       
       
            
print(x_augummented)      
print(x_augummented.shape)       
            
            
x_augummented = train_datagen.flow(
    x_augummented, y_augummented ,
    batch_size=argumet_size,
    shuffle = False
).next()[0]                 


print(x_augummented[0])     

print(x_train.shape)        

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)


print(x_train.shape,x_augummented.shape)


x_train = np.concatenate((x_train,x_augummented))        
y_train = np.concatenate((y_train,y_augummented))
print(x_train.shape,y_train.shape)         


es = EarlyStopping(monitor= 'val_loss' , mode = 'min' , patience= 10 , restore_best_weights=True , verbose=1    ) 


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#2 모델구성 

model = Sequential()
model.add(Conv2D(  70   ,(2,2),input_shape = (28,28,1) ))
model.add(Conv2D( filters = 6  ,  kernel_size = (3,3), strides= 2 , padding='valid'))
model.add(Conv2D( 20 ,(4,4),activation='relu',padding='same'))
model.add(MaxPooling2D())
model.add(MaxPooling2D())
model.add(Conv2D( 13 ,(2,2)))
model.add(Conv2D( 20 ,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(units= 84 ))
model.add(Dense(units= 6 ,input_shape = (8,)))
model.add(Dense(97,activation='relu'))
model.add(Dense(9))
model.add(Dense(81,activation='relu'))
model.add(Dense(10,activation= 'softmax'))



#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['acc'] )
model.fit(x_train, y_train, epochs = 100000 , batch_size= 1000 , verbose = 1 , validation_split= 0.2 , callbacks=[es] )



#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ',loss[0])
print('acc = ',loss[1])
# 오류가 나는 이유 // Shapes (32,) and (32, 27, 27, 10) are incompatible = 호환되지 않는다. 32, 와 32, 27, 27, 10 가 호환 X
# (32,) = 1차원 (32,27,27,10) = 4차원 이라서 오류가 발생한다.



y_test = np.argmax(y_test,axis=1)
y_predict = np.argmax(model.predict(x_test),axis=1)

def ACC(y_test,y_predict) : 
    return accuracy_score(y_test,y_predict)
acc = ACC(y_test,y_predict)

print("Acc =",acc)






# Epoch 309: early stopping
# 313/313 [==============================] - 1s 1ms/step - loss: 0.0773 - acc: 0.9763
# loss =  0.07726940512657166
# acc =  0.9763000011444092
# 시간 =  259.7592794895172


# Epoch 312: early stopping
# 313/313 [==============================] - 1s 1ms/step - loss: 0.0639 - acc: 0.9824
# loss =  0.06392145156860352
# acc =  0.9824000000953674
# 시간 =  261.86352729797363


# Epoch 80: early stopping
# 313/313 [==============================] - 1s 1ms/step - loss: 0.0672 - acc: 0.9797
# loss =  0.06718617677688599
# acc =  0.9797000288963318
# 시간 =  40.695334911346436
# 313/313 [==============================] - 0s 828us/step
# Acc = 0.9797



# 증폭
# Epoch 11: early stopping
# 313/313 [==============================] - 1s 1ms/step - loss: 0.8559 - acc: 0.6978
# loss =  0.8558865189552307
# acc =  0.6977999806404114
# 313/313 [==============================] - 0s 833us/step
# Acc = 0.6978