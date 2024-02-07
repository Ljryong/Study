import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential , Model , load_model
from keras.layers import Dense , Input ,concatenate , Conv1D ,Flatten , LSTM , ConvLSTM1D ,MaxPooling1D
from keras.callbacks import EarlyStopping , ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,r2_score , f1_score
from sklearn.preprocessing import LabelEncoder ,MaxAbsScaler , MinMaxScaler ,RobustScaler ,StandardScaler


#1 데이터
path = 'c:/_data/sihum//'
# csv 파일 가져오면서 , 제거
train1 = pd.read_csv(path + '삼성 240205.csv', encoding='cp949', thousands=',' ,index_col=0 )
train2 = pd.read_csv(path + '아모레 240205.csv',encoding='cp949',thousands=',' ,index_col=0)

# 데이터 수치화 
# train1['전일비'] = train1['전일비'].replace({'▼' : 0 , '▲' : 1 , ' ': 2})
# train2['전일비'] = train2['전일비'].replace({'▼' : 0 , '▲' : 1 , ' ': 2})

train1 = train1.drop(['외인비'],axis=1).drop(['신용비'],axis=1).drop(['전일비'],axis=1).drop(['개인'],axis=1).drop(['기관'],axis=1).drop(['외인(수량)'],axis=1)
train2 = train2.drop(['외인비'],axis=1).drop(['신용비'],axis=1).drop(['전일비'],axis=1).drop(['개인'],axis=1).drop(['기관'],axis=1).drop(['외인(수량)'],axis=1)

# train1 = train1[train1['개인'] != 0 ]
# train2 = train2[train2['개인'] != 0 ]

# train1['전일비'] = le.fit_transform(train1['전일비'])
# train2['전일비'] = le.fit_transform(train1['전일비'])

# 결측치 찾고 해결
print(train1.isna().sum())
print('================================================================')
print(train2.isna().sum())

print(type(train1))         # <class 'pandas.core.frame.DataFrame'>
print(type(train2))         # <class 'pandas.core.frame.DataFrame'>


# train1 = train1.fillna(train1['거래량'].mean())
# train1 = train1.fillna(train1['금액(백만)'].mean())
# train2 = train2.fillna(train2['거래량'].mean())
# train2 = train2.fillna(train2['금액(백만)'].mean())

# train1 = train1.dropna()
# train2 = train2.dropna()

train1 = train1.fillna(train1.ffill)          # 앞방향으로 똑같은거 채우기
train2 = train2.fillna(train2.ffill)  
# train1 = train1.fillna(train1.bfill)          # 뒷방향으로 똑같은거 채우기
# train2 = train2.fillna(train2.bfill) 

train1 = train1.sort_values(['일자'],ascending=True)
train2 = train2.sort_values(['일자'],ascending=True)

# train1_test = train1['시가'][-5:]
# train2_test = train2['종가'][-5:]

timesteps = 5
adds = 2
# 나누기
def split_xy(dataFrame, cutting_size, y_behind_size,  y_column):
    xs = []
    ys = [] 
    for i in range(len(dataFrame) - cutting_size - y_behind_size):
        x = dataFrame[i : (i + cutting_size)]
        y = dataFrame[i + cutting_size + y_behind_size : (i + cutting_size + y_behind_size + 1) ][y_column]
        xs.append(x)
        ys.append(y)
    return (np.array(xs), np.array(ys))


train1 = train1[train1.index > '2020/11/12']
train2 = train2[train2.index > '2020/11/12']

timestep = 5
add = 2
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

# 샘플 추출 x
compare_predict_size = 10
samsung_sample_x = x1[-compare_predict_size:]
amore_sample_x = x2[-compare_predict_size :]
# 샘플 추출 y
samsung_sample_y = np.append(y1[-compare_predict_size + 2:], ["24/02/06 시가","24/02/07 시가"]) 
amore_sample_y = np.append(y2[-compare_predict_size + 2:], ["24/02/06 종가","24/02/07 종가"])
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
                                                                                  test_size=0.3 , random_state= 980 , shuffle=True )


# print(x1_train.shape)       # (992, 6, 16)
# print(x2_train.shape)       # (992, 6, 16)
# print(x1_test.shape)        # (426, 6, 16)
# print(x2_test.shape)        # (426, 6, 16)

x1_train = x1_train.reshape(-1,50)
x2_train = x2_train.reshape(-1,50)
x1_test = x1_test.reshape(-1,50)
x2_test = x2_test.reshape(-1,50)
r_samsung_sample_x = samsung_sample_x.reshape(samsung_sample_x.shape[0], samsung_sample_x.shape[1] * samsung_sample_x.shape[2])
r_amore_sample_x = amore_sample_x.reshape(amore_sample_x.shape[0], amore_sample_x.shape[1] * amore_sample_x.shape[2])


# scaler 사용
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x1_train)
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)
r_samsung_sample_x = scaler.transform(r_samsung_sample_x)
scaler.fit(x2_train)
x2_test = scaler.fit_transform(x2_test)
x2_train = scaler.transform(x2_train)
r_amore_sample_x = scaler.transform(r_amore_sample_x)

x1_train = x1_train.reshape(-1,5,10)
x2_train = x2_train.reshape(-1,5,10)
x1_test = x1_test.reshape(-1,5,10)
x2_test = x2_test.reshape(-1,5,10)

# print()

samsung_sample_x = r_samsung_sample_x.reshape(-1, samsung_sample_x.shape[1], samsung_sample_x.shape[2])
amore_sample_x = r_amore_sample_x.reshape(-1, amore_sample_x.shape[1], amore_sample_x.shape[2])

# print(samsung_sample_x.shape)           # (10, 6, 16, 1)
# print(amore_sample_x.shape)             # (10, 6, 16, 1)

model = load_model('C:/_data/sihum/sihum_2.h5')


#4 평가, 예측
print(x1_test.shape,x2_test.shape)  # (423, 6, 16) , (423, 6, 16)

loss = model.evaluate([x1_test,x2_test],[y1_test,y2_test])
pre = model.predict([x1_test,x2_test])

r21 = r2_score(y1_test,pre[0])
r22 = r2_score(y2_test,pre[1])


# for i in range(5):
#     print('실제' , train1_test[i] , '예측', pre[0][i] )

# for i in range(5):
#     print('실제' , train2_test[i] , '예측', pre[1][i] )

print("="*100)
sample_dataset_y = [samsung_sample_y,amore_sample_y]
sample_predict_x = model.predict([samsung_sample_x,amore_sample_x])


for i in range(len(sample_dataset_y)):
    if i == 0 :
        print("\t\tSAMSUNG\t시가")
    else:
        print("="*100)
        print("\t\tAMORE\t종가")
    for j in range(compare_predict_size):
        print(f"\tD-{compare_predict_size - j  - 1}: {sample_dataset_y[i][j]}\t예측값 {np.round(sample_predict_x[i][j])}\t")
print("="*100)


print('loss',loss)
print('시가 r2',r21)
print('종가 r2',r22)

