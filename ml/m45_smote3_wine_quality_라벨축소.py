### 라벨을 관리할 수 있을 때 사용한다

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

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0 , k_neighbors = 3 )
x_train , y_train = smote.fit_resample(x_train,y_train)      

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

# SMOTE 미사용
# acc 0.6854545454545454
# F1 0.38804344220470954
    
# SMOTE 사용
# acc 0.6563636363636364
# F1 0.4124670695218978

