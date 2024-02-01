# 5일분(720행)을 훈련시켜서 하루(144행) 뒤를 예측한다. 


from keras.models import Sequential
from keras.layers import Dense , Dropout , SimpleRNN , LSTM ,GRU ,Conv1D , Flatten
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
import time 

#1 데이터
start = time.time()
path = 'c:/_data/kaggle/jena//'

xy = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

# print(xy.shape)             # (420551, 14)


x  = xy
y = xy['T (degC)']

timestep = 720

def split(xy,timestep,col) : 
    x=[]
    y=[]
    for i in range(len(xy) - timestep-144) :
        a,b = xy[i : (i+timestep)] , xy.iloc[i+timestep+144][col]           # iloc는 pandas가 : 을 잘 인식하지 못해서 인식하기 위해 쓰는것이다.
                                                                # pandas 인걸 확인하는 법은 읽어올때 pandas로 읽어서 그런것이고 이것 말고도 type으로 찍어서 확인이 가능하다.
        x.append(a)
        y.append(b)
    return np.array(x) , np.array(y)
# iloc = index 를 제외한 수치(값)들만을 뽑아내준다. 
x , y = split(xy,timestep,'T (degC)')


# print(x.shape,y.shape)       # (420541, 10, 14) (420541,)

end = time.time()

# print(test.shape)       # (420551,)

x_train , x_test , y_train, y_test = train_test_split(x,y, test_size = 0.3 , random_state = 0  , shuffle = True )
es = EarlyStopping(monitor='val_loss'  , mode = 'min' , patience= 50 , restore_best_weights=True , verbose= 1  )

#2 모델구성
model = Sequential()
model.add(Conv1D(6,2,input_shape = (720,14),activation='relu', padding='same'))
model.add(LSTM(3,activation='relu'))
model.add(Dense(1))


#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer='adam' , metrics=['mse'])
model.fit(x_train,y_train, epochs= 100 , batch_size= 500 , validation_split=0.2 , callbacks=[es])


#4 평가, 예측
loss = model.evaluate(x_test,y_test)
predict = model.predict(x_test)


print('loss = ' ,loss)
print('결과 = ' ,predict)
print('시간' , end - start)


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



# Epoch 10: early stopping
# 24/24 [==============================] - 3s 135ms/step - loss: nan - acc: 0.3333
# 24/24 [==============================] - 3s 130ms/step
# loss nan
# acc 0.3333333432674408


