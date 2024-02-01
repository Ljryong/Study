from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense , Dropout , Conv2D , MaxPooling2D , Flatten , LSTM , Conv1D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

(x_train,y_train) , (x_test,y_test) = cifar10.load_data()

x_train= x_train/255.
x_test= x_test/255.


train_datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip=True,           # 수평 뒤집기
    vertical_flip=True,             # 수직 뒤집기
    width_shift_range=0.2,          # 가로이동 비율
    height_shift_range=0.2,         # 세로이동 비율
    rotation_range=30,              # 회전 각도 조절
    zoom_range=0.2,                 # 축소, 확대 비율 조절
    shear_range=0.2,                # 찌끄러뜨리거나 , 누르기 
    fill_mode='nearest'
    
)

# argumet_size = 10000

# randidx = np.random.randint(x_train.shape[0], size = argumet_size)               # 랜덤한 인트값을 뽑는다.


# # print(randidx)                                  # [35727 44701 23616 ... 43784 46178 41121]
# # print(np.min(randidx), np.max(randidx) )        # 1 49997


# x_augummented = x_train[randidx].copy()          # 원데이터에 영향을 미치지 않기 위해서 .copy() 를 쓴다

# y_augummented = y_train[randidx].copy()

# # print(x_augummented)
# # print(x_augummented.shape)           # (10000, 32, 32, 3)
# # print(y_augummented)
# # print(y_augummented.shape)           # (10000, 1)



# x_augummented = x_augummented.reshape(10000,96,32)
#             # = x_augummented.reshape(-1,28,28,1)
#             # = x_augummented.reshape(x_augummented[0],x_augummented[1],x_augummented[2],1)
            
# print(x_augummented)       
# print(x_augummented.shape)      # (10000, 32, 32, 3)
            
            
# x_augummented = train_datagen.flow(
#     x_augummented, y_augummented ,          # x,y 가 다 있어야 뭐가 뭔지 판단할 수 있다.
#     batch_size=argumet_size,
#     shuffle = False
# ).next()[0]                    # .next 뒤에 [0] 을 쓰면 x 값만 나온다.


# print(x_augummented[0])         # [0]은 x [1]은 y

# print(x_train.shape)                # (60000, 28, 28)

x_train = x_train.reshape(50000,96,32)
x_test = x_test.reshape(10000,96,32)


# print(x_train.shape,x_augummented.shape)


# x_train = np.concatenate((x_train,x_augummented))              #concatenate: 사슬처럼 엮다
# y_train = np.concatenate((y_train,y_augummented))
# print(x_train.shape,y_train.shape)          # (60000, 32, 32, 3) (60000, 1)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)
print(y_test)

es = EarlyStopping(monitor= 'val_loss' , mode = 'min' , patience = 30 , restore_best_weights=True , verbose= 1 )

#2 모델구성 
model = Sequential()
model.add(Conv1D(50,2,input_shape = (96,32)))
model.add(Conv1D(5,2))
model.add(Flatten())
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(27))
model.add(Dense(10,activation='softmax'))

model.summary()

#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train,epochs = 100 ,batch_size= 10000 , verbose= 2 , callbacks=[es] , validation_split=0.2 )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1)
y_test = np.argmax(y_test,axis=1)

print('loss',loss[0])
print('loss_acc',loss[1])
print('acc', accuracy_score(y_test,y_predict) )






# 증폭
# Epoch 37: early stopping
# 313/313 [==============================] - 0s 1ms/step - loss: 1.3284 - acc: 0.5208
# 313/313 [==============================] - 0s 748us/step
# loss 1.3283514976501465
# loss_acc 0.520799994468689
# acc 0.5208


# LSTM
# Epoch 226: early stopping
# 313/313 [==============================] - 1s 3ms/step - loss: 1.3566 - acc: 0.5144
# 313/313 [==============================] - 1s 2ms/step
# loss 1.3565789461135864
# loss_acc 0.5144000053405762
# acc 0.5144


# Conv1D
# Epoch 100/100
# 4/4 - 0s - loss: 1.6944 - acc: 0.4217 - val_loss: 1.7415 - val_acc: 0.4031 - 232ms/epoch - 58ms/step
# 313/313 [==============================] - 1s 1ms/step - loss: 1.7169 - acc: 0.4080
# 313/313 [==============================] - 0s 712us/step
# loss 1.7168841361999512
# loss_acc 0.40799999237060547
# acc 0.408


