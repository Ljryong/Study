import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential , Model
from keras.layers import Dense , Input ,concatenate , Conv1D ,Flatten , LSTM
from keras.callbacks import EarlyStopping , ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,r2_score , f1_score
from sklearn.preprocessing import LabelEncoder ,MaxAbsScaler , MinMaxScaler ,RobustScaler ,StandardScaler


#1 데이터
path = 'c:/_data/sihum//'
# csv 파일 가져오면서 , 제거
train1 = pd.read_csv(path + '삼성240205.csv', index_col=0 , encoding='cp949', thousands=',' )
train2 = pd.read_csv(path + '아모레240205.csv', index_col=0 ,encoding='cp949',thousands=',' )

# 데이터 수치화 
train1['전일비'] = train1['전일비'].replace({'▼' : 0 , '▲' : 1 , ' ': 2})
train2['전일비'] = train2['전일비'].replace({'▼' : 0 , '▲' : 1 , ' ': 2})

# train1['전일비'] = le.fit_transform(train1['전일비'])
# train2['전일비'] = le.fit_transform(train1['전일비'])

# 결측치 찾고 해결
print(train1.isna().sum())
print('================================================================')
print(train2.isna().sum())

print(type(train1))         # <class 'pandas.core.frame.DataFrame'>
print(type(train2))         # <class 'pandas.core.frame.DataFrame'>


train1 = train1.fillna(train1['거래량'].mean())
train1 = train1.fillna(train1['금액(백만)'].mean())
train2 = train2.fillna(train2['거래량'].mean())
train2 = train2.fillna(train2['금액(백만)'].mean())



# train1 = train1.dropna()
# train2 = train2.dropna()

# train1 = train1.fillna(train1.ffill)          # 앞방향으로 똑같은거 채우기
# train2 = train2.fillna(train2.ffill)  
# train1 = train1.fillna(train1.bfill)          # 뒷방향으로 똑같은거 채우기
# train2 = train2.fillna(train2.bfill) 

# 나누기
timestep = 6
add = 1
def split(train1,train2,timestep,col1,col2,add) : 
    x=[]
    y=[]
    z=[]
    f=[]
    for i in range(len(train1) - timestep - add) :
        a , b = train1[i : i+timestep] , train1.iloc[i+timestep+add][col1]
        # print(type(train1))         # <class 'pandas.core.frame.DataFrame'>
        # print(type(train2))         # <class 'pandas.core.frame.DataFrame'>
        # print(a)
        # print(b)
        x.append(a)
        y.append(b)
    for i in range(len(train2) - timestep - add) : 
        c,d = train2[i : i+timestep ] , train2.iloc[i+timestep+add][col2]
        z.append(c)
        f.append(d)
    return np.array(x) , np.array(y) , np.array(z) , np.array(f)




x1 , y1 , x2, y2 = split(train1,train2,timestep,'시가','종가',add)

# iloc = index 를 제외한 수치(값)들만을 뽑아내준다. 



# 슬라이싱으로 데이터 범위 정하기
train1 = np.array(train1)       
train2 = np.array(train2)

end_row = 1418
x1 = x1[:end_row,:].astype(np.float32)
x2 = x2[:end_row,:].astype(np.float32)
y1 = y1[:end_row].astype(np.float32)
y2 = y2[:end_row].astype(np.float32)




print(x1)       # (1418, 6, 16)
print(x2)       # (1418, 6, 16)
print(y1)       # (1418,)
print(y2)       # (1418,)

# x1 = train1[train1['시가'] <= 100000]
# x1 = train1[train1['시가'] >= 40000]

# x2 = train2[train2['시가'] <= 450000]




x1_train , x1_test , x2_train , x2_test , y1_train , y1_test ,y2_train,y2_test = train_test_split(x1,x2,y1,y2,
                                                                                  test_size=0.3 , random_state= 730320 , shuffle=True )


print(x1_train.shape)       # (992, 6, 16)
print(x2_train.shape)       # (992, 6, 16)
print(x1_test.shape)        # (426, 6, 16)
print(x2_test.shape)        # (426, 6, 16)

x1_train = x1_train.reshape(-1,96)
x2_train = x2_train.reshape(-1,96)
x1_test = x1_test.reshape(-1,96)
x2_test = x2_test.reshape(-1,96)


# scaler 사용
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

x1_train = scaler.fit_transform(x1_train)
x2_train = scaler.fit_transform(x2_train)
x1_test = scaler.fit_transform(x1_test)
x2_test = scaler.fit_transform(x2_test)

x1_train = x1_train.reshape(-1,6,16)
x2_train = x2_train.reshape(-1,6,16)
x1_test = x1_test.reshape(-1,6,16)
x2_test = x2_test.reshape(-1,6,16)

print('데이터 전처리')

#2 모델구성
#2-1 모델1
in1 = Input(shape=(6,16))
d1 = LSTM(64,activation='swish')(in1)
d2 = Dense(32,activation='swish')(d1)
d3 = Dense(64,activation='swish')(d2)
d4 = Dense(32,activation='swish')(d3)
d5 = Dense(16,activation='swish')(d4)
d6 = Dense(8,activation='swish')(d5)
d7 = Dense(64,activation='swish')(d6)
d8 = Dense(32,activation='swish')(d7)
d9 = Dense(16,activation='swish')(d8)
d10 = Dense(32,activation='swish')(d9)
d11 = Dense(64,activation='swish')(d10)
d12 = Dense(32,activation='swish')(d11)
d13 = Dense(16,activation='swish')(d12)
out1 = Dense(64,activation='swish')(d13)

#2-2 모델2
in12 = Input(shape=(6,16))
d31 = Conv1D(64,2,activation='swish')(in12)
d32 = Conv1D(32,2,activation='swish')(d31)
d33 = Conv1D(64,2,activation='swish')(d32)
d34 = Conv1D(16,2,activation='swish')(d33)
d35 = Conv1D(32,2,activation='swish')(d34)
f1 = Flatten()(d35)
d36 = Dense(16,activation='swish')(f1)
d37 = Dense(32,activation='swish')(d36)
d38 = Dense(64,activation='swish')(d37)
d39 = Dense(16,activation='swish')(d38)
d40 = Dense(16,activation='swish')(d39)
d41 = Dense(32,activation='swish')(d40)
d42 = Dense(64,activation='swish')(d41)
d43 = Dense(32,activation='swish')(d42)
d44 = Dense(16,activation='swish')(d43)
out2 = Dense(32,activation='swish')(d44)

#2-3 모델결합
merge = concatenate([out1,out2])
mgd1 = Dense(64,activation='swish')(merge)
mgd2 = Dense(32,activation='swish')(mgd1)
mgd3 = Dense(64,activation='swish')(mgd2)
mgd4 = Dense(16,activation='swish')(mgd3)
last = Dense(1,activation='swish')(mgd4)

last2 = Dense(1,activation='swish')(mgd4)


model = Model(inputs = [in1,in12] , outputs = [last,last2])

#3 컴파일,훈련
model.compile(loss = 'mae' ,optimizer='adam' ) # , metrics=['mae'] 
es = EarlyStopping(monitor='val_loss' , mode = 'min' , restore_best_weights=True , patience= 500 , verbose=1 )
# modelcheckpoint
# filepath = 'c:/_data/_save/MCP//'
# mcp = ModelCheckpoint(monitor='val_loss' , mode = 'min' , save_best_only=True , verbose= 1 , filepath=filepath  )
model.fit([x1_train,x2_train],[y1_train,y2_train] ,epochs= 10000000 , batch_size= 300 , validation_split=0.2 , callbacks=[es], verbose=2 )
model.save("c:/_data/sihum/sihum_1.h5")

#4 평가, 예측
loss = model.evaluate([x1_test,x2_test],[y1_test,y2_test])
pre = model.predict([x1_test,x2_test])

r21 = r2_score(y1_test,pre[0])
r22 = r2_score(y2_test,pre[1])


for i in range(5):
    print('실제' , y1_test[i] , '예측', pre[0][i] )

for i in range(5):
    print('실제' , y2_test[i] , '예측', pre[1][i] )



print('loss',loss)
# print('시가',pre[0][0])
# print('종가',pre[1][0])
print('시가 r2',r21)
print('종가 r2',r22)




# Epoch 602: early stopping
# 14/14 [==============================] - 0s 6ms/step - loss: 19363.1465 - dense_27_loss: 12515.0947 - dense_28_loss: 6848.0503
# 14/14 [==============================] - 0s 3ms/step
# 실제 73600.0 예측 [81118.42]
# 실제 68900.0 예측 [75797.89]
# 실제 61600.0 예측 [59931.133]
# 실제 60300.0 예측 [53672.457]
# 실제 59900.0 예측 [59339.562]
# 실제 211500.0 예측 [210681.02]
# 실제 167000.0 예측 [169071.22]
# 실제 127500.0 예측 [127170.04]
# 실제 161000.0 예측 [158380.67]
# 실제 130800.0 예측 [124379.23]
# loss [19363.146484375, 12515.0947265625, 6848.05029296875]
# 시가 r2 0.6152906483620448
# 종가 r2 0.9141168757637294




# 실제 73600.0 예측 [79894.35]
# 실제 68900.0 예측 [75772.24]
# 실제 61600.0 예측 [60049.816]
# 실제 60300.0 예측 [59498.668]
# 실제 59900.0 예측 [59213.562]
# 실제 211500.0 예측 [204051.2]
# 실제 167000.0 예측 [167838.95]
# 실제 127500.0 예측 [128112.984]
# 실제 161000.0 예측 [154899.67]
# 실제 130800.0 예측 [127620.66]
# loss [25847.0390625, 19570.0859375, 6276.95263671875]
# 시가 r2 0.4567748031077147
# 종가 r2 0.9761658356141811

