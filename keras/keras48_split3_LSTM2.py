import numpy as np
from keras.models import Sequential
from keras.layers import Dense , SimpleRNN , LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


#1 데이터
a = np.array(range(1,101))
x_predict = np.array(range(96,106))
size = 5            # 자르는 사이즈(time steps)             x 데이터는 4개 , y 데이터는 1개

def split_x(dataset,size) :         # dataset = a
    aaa=[]                          # aaa 리스트 만들기
    for i in range(len(dataset) - size+1):          # range(1~10 에서 5빼고 1 더하기 ) = 반복 횟수(행의 갯수를 알 수 있음)
                                                    # dataset의 길이에서 size를 뺀 만큼 반복합니다(6 = 행의 갯수 ). 이렇게 함으로써 생성할 부분 시계열 데이터의 개수(행)를 결정합니다
        subset = dataset[ i : ( i + size ) ]        # range(1:11)[i:(i+5)] i가 0 일때 0 : 0 + 5 = 1,2,3,4,5 를 뜻한다.
        aaa.append(subset)      # 이어 붙이기        # for문(반복문)이어서 반복된걸 append(이어 붙여준다)
    return np.array(aaa)        # 이어붙인걸 리스트 형태로 반환해서 넣어줄 틀을 만들어준다, 틀이 없으면 들어가지 않는다.

bbb = split_x(a,size)       # 위에 정의한 split_x 안에 값을 넣고 값을 함수의 형식으로 치환하기
print(bbb)                  

print(bbb.shape)            # (96, 5)

x = bbb[: , :-1]            # 모든 행에서 마지막 열 빼고 뽑아주세요
y = bbb[: , -1]             # 모든 행에서 마지막 열만 뽑아주세요
print(x , y)                # 분리하는 이유는 x 

print(x.shape,y.shape)      # (96, 4) (96,)

size = 4

def split_xpre(ccc,size) : 
    ccc=[]
    for i in range(len(x_predict)-size+1):
        subset = ccc[i : (i + size)]
        ccc.append(subset)
    return np.array(ccc)    

ccc = split_x(x_predict,size)           # 얘가 예측할 값

print(ccc)
print(ccc.shape)            # (7, 4)
ccc = ccc.reshape(-1,1,4)
x = x.reshape(-1,1,4)
# x_predict = x_predict.reshape(7,4,1)


# x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state= 0 ,stratify=y )
# train_test_split 을 쓰면 y값의 데이터가 너무 적어져서 사용할 수 없다고 뜸

es = EarlyStopping(monitor='loss' , mode='min' , patience= 100 , restore_best_weights=True , verbose=1 )

#2 모델구성
model = Sequential()
model.add(LSTM(128,input_shape=(1,4),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))


#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer='adam' , metrics= ['mse'] )
# model.fit(x_train,y_train,epochs= 100000 , batch_size=10  , callbacks=[es] , validation_split=0.2 , verbose=1  )
model.fit(x,y,epochs= 100000 , batch_size=10  , callbacks=[es] , verbose=1  )       # , validation_split=0.2 

#4 평가, 예측
loss = model.evaluate(x,y)
# loss = model.evaluate(x_test,y_test)
y_predict = model.predict(ccc)

print('loss = ' , loss)
print('결과 = ' , y_predict)


# (4,1)
# loss =  [5.6260538258356974e-05, 5.6260538258356974e-05]
# 결과 =  [[100.01397 ]
#  [101.01667 ]
#  [102.01952 ]
#  [103.022545]
#  [104.02571 ]
#  [105.02907 ]
#  [106.032585]]


# (2,2)
# loss =  [3.9851467590779066e-05, 3.9851467590779066e-05]
# 결과 =  [[100.01248 ]
#  [101.0163  ]
#  [102.02048 ]
#  [103.02501 ]
#  [104.02985 ]
#  [105.035034]
#  [106.04054 ]]

# (1,4)
# loss =  [0.00012018494453513995, 0.00012018494453513995]
# 결과 =  [[100.01135 ]
#  [101.01429 ]
#  [102.01743 ]
#  [103.02072 ]
#  [104.024216]
#  [105.02786 ]
#  [106.0317  ]]

