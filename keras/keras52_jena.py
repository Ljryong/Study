from keras.models import Sequential
from keras.layers import Dense , Dropout , SimpleRNN , LSTM ,GRU
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

#1 데이터
path = 'c:/_data/kaggle/jena//'

xy = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

# print(xy.shape)             # (420551, 14)


x  = xy
y = xy['T (degC)']

size = 10

def split(xy,size,col) : 
    x=[]
    y=[]
    for i in range(len(xy) - size) :
        a,b = xy[i : (i+size)] , xy.iloc[i+size][col]           # iloc는 pandas가 : 을 잘 인식하지 못해서 인식하기 위해 쓰는것이다.
                                                                # pandas 인걸 확인하는 법은 읽어올때 pandas로 읽어서 그런것이고 이것 말고도 type으로 찍어서 확인이 가능하다.
        x.append(a)
        y.append(b)
    return np.array(x) , np.array(y)
# iloc = index 를 제외한 수치(값)들만을 뽑아내준다. 
x , y = split(xy,size,'T (degC)')


# print(x.shape,y.shape)       # (420541, 10, 14) (420541,)



# print(test.shape)       # (420551,)



x_train , x_test , y_train, y_test = train_test_split(x,y, test_size = 0.3 , random_state = 0  , shuffle = True )
es = EarlyStopping(monitor='val_loss'  , mode = 'min' , patience= 50 , restore_best_weights=True , verbose= 1  )

#2 모델구성
model = Sequential()
model.add(GRU(128,input_shape = (10,14),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))


#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer='adam' , metrics=['mse'])
model.fit(x_train,y_train, epochs= 1000 , batch_size= 10000 , validation_split=0.2 , callbacks=[es])


#4 평가, 예측
loss = model.evaluate(x_test,y_test)
predict = model.predict(x_test)

print('loss = ' ,loss)
print('결과 = ' ,predict)


# LSTM
# loss =  [0.44846034049987793, 0.44846034049987793]
# 결과 =  [[18.965517 ]
#  [-1.9998345]
#  [ 0.7080312]
#  ...
#  [16.786592 ]
#  [-6.1850452]
#  [ 5.322441 ]]


# GRU
# loss =  [0.048605289310216904, 0.048605289310216904]
# 결과 =  [[18.904936 ]
#  [-1.8903892]
#  [ 1.4161297]
#  ...
#  [16.741688 ]
#  [-5.7747936]
#  [ 5.4525313]]








