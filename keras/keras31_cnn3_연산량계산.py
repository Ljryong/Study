import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense , Conv2D , Flatten       # Flatten : 평평한

#1 데이터
(x_train , y_train), (x_test, y_test)  =  mnist.load_data()
print(x_train.shape , y_train.shape)    # (60000, 28, 28) (60000,)
print(x_test.shape , y_test.shape)      # (10000, 28, 28) (10000,)

print(x_train)
print(x_train[0])
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))
print(pd.value_counts(y_test))

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
# print(x_train.shape[0])         # 60000

# x_test = x_test.reshape(x_test[0],x_test[1],x_test[2],1)        # 위에 10000,28,28,1 이랑 같다. 이렇게 쓰는게 나중에 전처리하고 test가 달라졌을 때 좋을수도 있다.

print(x_train.shape , x_test.shape)     # (60000, 28, 28, 1) (10000, 28, 28, 1)



#2 모델구성
model = Sequential()
model.add(Conv2D(9,(2,2),input_shape = (28,28,1) ))        # 10 = 필터 / (2,2) = 커널 사이즈
#                              shape = (batch_size(model.fit에 들어가는 batch_size // 행이랑 똑같다),rows,columns,channels)
#                              shape = (batch_size,heights,widths,channels)

model.add(Conv2D(filters = 10,kernel_size = (3,3)))
model.add(Conv2D(15,(4,4)))
model.add(Flatten())
model.add(Dense(units=8))
model.add(Dense(units=7,input_shape = (8,)))
#                             shape = (batch_size , input_dim )

model.add(Dense(6))
model.add(Dense(10,activation= 'softmax'))

model.summary()

"""
(kernel*channel + bias) * filters

1번째 레이어 = (4*1+1)*9=45
2번째 레이어 = (9*9+1)*10=820
3번째 레이어 = (16*10+1)*15=2415


"""
'''
#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['acc'] )
model.fit(x_train,y_train , epochs = 100 , batch_size=32 , verbose=1 , validation_split= 0.2 )



#4 평가, 예측
result = model.evaluate(x_test, y_test)
print('loss = ',result[0])
print('acc = ',result[1])

# 오류가 나는 이유 // Shapes (32,) and (32, 27, 27, 10) are incompatible = 호환되지 않는다. 32, 와 32, 27, 27, 10 가 호환 X
# (32,) = 1차원 (32,27,27,10) = 4차원 이라서 오류가 발생한다.

'''



