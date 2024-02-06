import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential , Model , load_model
from keras.layers import Dense , Input ,concatenate , Conv1D ,Flatten , LSTM
from keras.callbacks import EarlyStopping , ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,r2_score , f1_score
from sklearn.preprocessing import LabelEncoder ,MaxAbsScaler , MinMaxScaler ,RobustScaler ,StandardScaler


#1 데이터
path = 'c:/_data/sihum//'
# csv 파일 가져오면서 , 제거
train1 = pd.read_csv(path + '삼성240205.csv', encoding='cp949', thousands=',' ,index_col=0 )
train2 = pd.read_csv(path + '아모레240205.csv',encoding='cp949',thousands=',' ,index_col=0)

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

train1 = train1.sort_values(['일자'],ascending=True)
train2 = train2.sort_values(['일자'],ascending=True)

train1_test = train1['시가'][-5:]
train2_test = train2['종가'][-5:]

print(train1_test)
print(train2_test)

train1 = train1[train1.index > '2018/05/04']
train2 = train2[train2.index > '2018/05/04']

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

# end_row = 8879-8
# end_row2 = 2933-8
# x1 = x1[end_row:,:]
# x2 = x2[end_row2:,:]
# y1 = y1[end_row:]
# y2 = y2[end_row2:]

# print(x1.shape)       # (1418, 6, 16)
# print(x2)       # (1418, 6, 16)
# print(y1.shape)       # (1418,)
# print(y2)       # (1418,)




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
scaler = MinMaxScaler()
# scaler = StandardScaler()
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


model = load_model('c:/_data/sihum/sihum.h5')



#4 평가, 예측
loss = model.evaluate([x1_test,x2_test],[y1_test,y2_test])
pre = model.predict([x1_test,x2_test])

r21 = r2_score(y1_test,pre[0])
r22 = r2_score(y2_test,pre[1])


for i in range(5):
    print('실제' , train1_test[i] , '예측', pre[0][i] )

for i in range(5):
    print('실제' , train2_test[i] , '예측', pre[1][i] )



print('loss',loss)
print('시가',pre[0][0])
print('종가',pre[1][0])
print('시가 r2',r21)
print('종가 r2',r22)