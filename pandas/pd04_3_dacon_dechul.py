from keras.models import Sequential
from keras.layers import Dense , Dropout , LeakyReLU , BatchNormalization
from keras.callbacks import EarlyStopping , ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder , LabelEncoder , MinMaxScaler
import datetime
from imblearn.over_sampling import SMOTE
import time
from sklearn.svm import LinearSVC

#1
path = 'c:/_data/dacon/dechul//'
train_csv = pd.read_csv(path+ 'train.csv' , index_col= 0)
test_csv = pd.read_csv(path+ 'test.csv' , index_col= 0)
submission_csv = pd.read_csv(path+ 'sample_submission.csv')

encoder = LabelEncoder()
encoder.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = encoder.transform(train_csv['주택소유상태'])

encoder.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])

encoder.fit(train_csv['대출등급'])
train_csv['대출등급'] = encoder.transform(train_csv['대출등급'])

date = datetime.datetime.now()
date = date.strftime('%m%d-%H%M')
path = 'c:/_data/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path , 'k28_11_', date , '_', filename ])

train_csv['주택소유상태'] = train_csv['주택소유상태'].replace({'MORTGAGE' : 0 , 'OWN' : 1 , 'RENT': 2 , 'ANY' : 0}).astype(float)
test_csv['주택소유상태'] = test_csv['주택소유상태'].replace({'MORTGAGE' : 0 , 'OWN' : 1 , 'RENT': 2}).astype(float)

train_csv['대출목적'] = train_csv['대출목적'].replace({'부채 통합' : 0 , '주택 개선' : 2 , '주요 구매': 4 , '휴가' : 9  
                                                     , '의료' : 5 , '자동차' : 6 , '신용 카드' : 1 , '기타' : 3 , '주택개선' : 8,
                                                      '소규모 사업' : 7 , '이사' :  12 , '주택': 10 , '재생 에너지' : 11 })
test_csv['대출목적'] = test_csv['대출목적'].replace({'부채 통합' : 0 , '주택 개선' : 2 , '주요 구매': 4 , '휴가' : 9 ,
                                             '의료' : 5 , '자동차' : 6 , '신용 카드' : 1 , '기타' : 3 , '주택개선' : 8,
                                             '소규모 사업' : 7 , '이사' :  12 , '주택': 10 , '재생 에너지' : 11 , 
                                             '결혼' : 2 })

# 결혼은 train에 없는 라벨이다. 그래서 12 로 두든 2로 두든 아니면 없애든 값이 좋은걸로 비교해보면 된다.
train_csv['대출기간'] = train_csv['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)
test_csv['대출기간'] = test_csv['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)

train_csv['근로기간'] = train_csv['근로기간'].replace({'10+ years' : 11 ,            # 10으로 둘지 그 이상으로 둘지 고민
                                                      '2 years' : 2 , '< 1 year' : 0.7 , '3 years' : 3 , '1 year' : 1 ,
                                                      'Unknown' : 0 , '5 years' : 5 , '4 years' : 4 , '8 years' : 8 ,
                                                      '6 years' : 6 , '7 years' : 7 , '9 years' : 9 , '10+years' : 11,
                                                      '<1 year' : 0.7 , '3' : 3 , '1 years' : 1 })
test_csv['근로기간'] = test_csv['근로기간'].replace({'10+ years' : 11 ,            # 10으로 둘지 그 이상으로 둘지 고민
                                                      '2 years' : 2 , '< 1 year' : 0.7 , '3 years' : 3 , '1 year' : 1 ,
                                                      'Unknown' : 0 , '5 years' : 5 , '4 years' : 4 , '8 years' : 8 ,
                                                      '6 years' : 6 , '7 years' : 7 , '9 years' : 9 , '10+years' : 11,
                                                      '<1 year' : 0.7 , '3' : 3 , '1 years' : 1 })

def outliers(data_out):
    quartile_1 , q2 , quartile_3 = np.percentile(data_out,[25,50,75])       # 25,50,75 퍼센트로 나눔
    print('1사 분위 :' , quartile_1 )
    print('q2 :' , q2 )
    print('3사 분위 :' , quartile_3 )
    iqr = quartile_3 - quartile_1                       
    # 이상치는 보통의 값을 벗어난 것인데 이상치는 엄청 크거나 엄청 작거나 둘중 하나이다
    # 이런걸 방지하기위해서 상위25%과 하쉬25%를 버리고 나머지 50%를 가져온다.
    # 가운데 데이터들은 보통 정상적인 데이터라고 판단(아닐수도 잇음) 
    print('iqr :' , iqr)
    lower_bound = quartile_1 - (iqr * 1.5)              
    # 1.5가 아니여도 되는데 통상 1.5가 제일 좋음
    # 로우 = 4 - (6 * 1.5) = 4 - 9 = -5 여기까지의 데이터를 이상치가 아니라고 판단한다
    upper_bound = quartile_3 + (iqr * 1.5)
    # 하이 = 10 + (6 * 1.5) = 10 + 9 = 19 여기까지의 데이터를 이상치가 아니라고 판단
    return np.where((data_out>upper_bound) | (data_out<lower_bound))        # | python 함수에서 or 이랑 같은 뜻이다
    # 2가지 조건중에 한개라도 만족하는걸 빼냄 19큰거 -5보다 작은걸 빼내라
    # 뽑으면 위치값 0 , 12 의 값이 이상치라고 나옴
outliers_loc = outliers(train_csv)
print('이상치의 위치 :' , outliers_loc)

import matplotlib.pyplot as plt
print(train_csv.info())
plt.boxplot(train_csv)
plt.show()

# train_csv = train_csv[train_csv['연간소득']<=1.8]
# test_csv = test_csv[test_csv['연간소득']<=1.8]

print(train_csv)          # (1, 14)

x = train_csv.drop(['대출등급' , '연간소득' ],axis = 1 )
y = train_csv['대출등급']
test_csv = test_csv.drop(['연간소득'],axis = 1)


# y = y.values.reshape(-1)       # (96294, 1)





x_train ,x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 ,train_size=0.8, random_state= 730501 , shuffle=True , stratify=y)    # 0 1502
es = EarlyStopping(monitor='val_loss', mode='min' , patience= 3000 , restore_best_weights=True , verbose= 1 )




# print(y_train.shape)            # (67405, 7) // print(y_train.shape) = output 값 구하는 법


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# test_csv = np.delete(test_csv,1,axis=1)
test_csv = scaler.transform(test_csv)        # test_csv도 같이 학습 시켜줘야 값이 나온다. 안해주면 소용이 없다.



#2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor

# model = RandomForestRegressor()
model = RandomForestClassifier()

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
print(x.shape)


# 쓰기 전 
# 0.92


# 사용 후
# 이상치가 컬럼 5에서 많은데 수치표현이 어려워 삭제
# 1사 분위 : 0.0
# q2 : 10.0
# 3사 분위 : 193875.0
# iqr : 193875.0
# 이상치의 위치 : (array([    0,     0,     1, ..., 96293, 96293, 96293], dtype=int64), array([0, 4, 0, ..., 0, 4, 9], dtype=int64))
# model.score 0.8211225920348928