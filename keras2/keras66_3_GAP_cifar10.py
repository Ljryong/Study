# Flatten 대신 GlobalAveragePooling 을 사용할 수 있다.
# Flatten 의 문제점이 있다 (너무 크다 = 연산량이 많다)
import numpy as np
from keras.datasets import mnist , cifar10
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense , Conv2D , Flatten  , MaxPooling2D , GlobalAveragePooling2D       # Flatten : 평평한
from keras.callbacks import ReduceLROnPlateau , EarlyStopping
from keras.optimizers import Adam

#1 데이터
(x_train , y_train), (x_test, y_test)  =  cifar10.load_data()
print(x_train.shape , y_train.shape)    # (60000, 28, 28) (60000,)
print(x_test.shape , y_test.shape)      # (10000, 28, 28) (10000,)

#2 모델구성
model = Sequential()
model.add(Conv2D(100,(2,2),input_shape = (32,32,3), padding='same' , strides= 1 ))        # 10 = 필터 / (2,2) = 커널 사이즈  // strides = 보폭의 크기 // padding = 'valid' 디폴트
#                              shape = (batch_size(model.fit에 들어가는 batch_size // 행이랑 똑같다),rows,columns,channels)
#                              shape = (batch_size,heights,widths,channels)
model.add(Conv2D(filters = 100,kernel_size = (2,2)))    # (None,4,4,100)
model.add(Conv2D(100,(2,2)))    # (None,3,3,100)
model.add(Conv2D(100,(2,2)))
model.add(Conv2D(100,(2,2)))    
model.add(Conv2D(100,(2,2)))    
# model.add(Flatten())
model.add(GlobalAveragePooling2D())     # (None,100)
model.add(Dense(units=50))              # (None,50) 5050
model.add(Dense(10,activation= 'softmax'))  #(None,10) 510

model.summary()

#3 컴파일, 훈련
es = EarlyStopping(monitor='val_loss' , mode='min' , restore_best_weights=True , patience= 10 , verbose=1 )
rlr = ReduceLROnPlateau(monitor='val_loss' , mode = 'min' , factor=0.7, patience=5    )

model.compile(loss = 'sparse_categorical_crossentropy' , optimizer = Adam(0.001) , metrics = ['acc'] )
model.fit(x_train,y_train , epochs = 10000 , batch_size=100 , verbose=1 , validation_split= 0.2 , callbacks=[es, rlr] )


#4 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss = ',result[0])
print('acc = ',result[1])

# 오류가 나는 이유 // Shapes (32,) and (32, 27, 27, 10) are incompatible = 호환되지 않는다. 32, 와 32, 27, 27, 10 가 호환 X
# (32,) = 1차원 (32,27,27,10) = 4차원 이라서 오류가 발생한다.

# Flatten
# loss =  1.925847053527832
# acc =  0.3393999934196472

# GlobalAveragePooling2D
# loss =  1.914096713066101
# acc =  0.31929999589920044
