### 44_1번을 rf 디폴트로 리폼

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score , f1_score
from sklearn.ensemble import RandomForestClassifier

#1 데이터

path = 'c:/_data/dacon/wine/'

train_csv = pd.read_csv(path + 'train.csv' , index_col= 0 )
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# print(train_csv.isna().sum())         # 없음
# print(test_csv.isna().sum())

# print(train_csv.shape)              # (5497, 13)

# print(train_csv)
la = LabelEncoder()
train_csv['type'] = la.fit_transform(train_csv['type'])
test_csv['type'] = la.fit_transform(test_csv['type'])

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']

####################################################################
# [실습] y의 클래스를 7개에서 5~3개로 줄여서 성능 비교
####################################################################
# 여기에다가 넣으면 됨
          # 이걸 써야 원본 데이터를 바꾸지 않고 사용할 수 있음

# 각 반복에서 피처를 추가로 제거하면서 성능 평가
# 내가 만든 것 맞는지는 모름
""" for i in range(5):
    
    a = y.copy()        
    # 이걸 for문 밖에다가 정의를 해주면 초기화가 되지않아서 그냥 class가 줄어든 상태로 시작함
    
    a[(a == i ) | (a == i+1 )] = i+2

    x_train , x_test , y_train , y_test = train_test_split(x,a, test_size = 0.2 , random_state= 12 , stratify=y , shuffle = True )
    
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #2 모델
    model = RandomForestClassifier(random_state = 980909 )

    #3 컴파일, 훈련
    model.fit(x_train , y_train )

    #4 평가
    acc = model.score(x_test,y_test)
    print('acc',acc)
    predict = model.predict(x_test)
    print('accuracy' , accuracy_score(y_test,predict) ) """

###################################################################

a = y.copy()    

# 각 반복에서 피처를 추가로 제거하면서 성능 평가 
# 선생님 for 문
for i,v in enumerate(a):
    
    if v <=4:
        y[i] = 0
    elif v==5:
        y[i] = 1
    elif v==6:
        y[i] = 2
    elif v==7:
        y[i] = 3
    elif v==8:
        y[i] = 4
    else :
        y[i] = 5

x_train , x_test , y_train , y_test = train_test_split(x,a, test_size = 0.2 , random_state= 12 , stratify=y , shuffle = True )

scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2 모델
model = RandomForestClassifier(random_state = 980909 )

#3 컴파일, 훈련
model.fit(x_train , y_train )

#4 평가
acc = model.score(x_test,y_test)
print('acc',acc)
predict = model.predict(x_test)
# print('accuracy' , accuracy_score(y_test,predict) )
print('F1' ,f1_score(y_test,predict,average='macro') )

# acc 0.6854545454545454
# F1 0.38804344220470954
    
####################################################################

