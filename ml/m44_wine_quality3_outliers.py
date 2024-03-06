# 아웃라이어 확인
# 아웃라이어 처리
# 44_1 , 44_2 든 수정해서 만들기

### 44_1번을 rf 디폴트로 리폼

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
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
y = train_csv['quality']-3

# 이상치 for문
# def outliers(data_out):
#     quartile_1 , q2 , quartile_3 = np.percentile(data_out,[25,50,75])       # 25,50,75 퍼센트로 나눔
#     print('1사 분위 :' , quartile_1 )
#     print('q2 :' , q2 )
#     print('3사 분위 :' , quartile_3 )
#     iqr = quartile_3 - quartile_1                       
#     # 이상치는 보통의 값을 벗어난 것인데 이상치는 엄청 크거나 엄청 작거나 둘중 하나이다
#     # 이런걸 방지하기위해서 상위25%과 하쉬25%를 버리고 나머지 50%를 가져온다.
#     # 가운데 데이터들은 보통 정상적인 데이터라고 판단(아닐수도 잇음) 
#     print('iqr :' , iqr)
#     lower_bound = quartile_1 - (iqr * 1.5)              
#     # 1.5가 아니여도 되는데 통상 1.5가 제일 좋음
#     # 로우 = 4 - (6 * 1.5) = 4 - 9 = -5 여기까지의 데이터를 이상치가 아니라고 판단한다
#     upper_bound = quartile_3 + (iqr * 1.5)
#     # 하이 = 10 + (6 * 1.5) = 10 + 9 = 19 여기까지의 데이터를 이상치가 아니라고 판단
#     return np.where((data_out>upper_bound) | (data_out<lower_bound))        # | python 함수에서 or 이랑 같은 뜻이다
#     # 2가지 조건중에 한개라도 만족하는걸 빼냄 19큰거 -5보다 작은걸 빼내라
#     # 뽑으면 위치값 0 , 12 의 값이 이상치라고 나옴
# outliers_loc = outliers(x)
# print('이상치의 위치 :' , outliers_loc)

# import matplotlib.pyplot as plt
# plt.boxplot(x)
# plt.show()

# print(x.columns)
# ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
#        'pH', 'sulphates', 'alcohol', 'type']

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination= .3 )        

outliers.fit(x)
results = outliers.predict(x)   

# 이상치로 분류된 샘플의 인덱스를 식별
outlier_indices = (results == -1)

# 이상치를 제거한 데이터셋 생성
x_clean = x[~outlier_indices]           # ~ 는 반전 시키는 것, True를 False로 False를 True로 반대로 바꾸는 것

print(x_clean)

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.2 , random_state= 12 , stratify=y , shuffle = True )

scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators' : 1000, 
              'learning_rate' : 0.5,
              'max_depth': 10,               # 트리 깊이
              'gamma' : 0,
              'min_child_weight' : 3,       # 드랍 아웃 개념
              'subsample' : 0.4,
              'colsample_bytree' : 0.8,
              'colsample_bylevel' : 0.7,
              'colsample_bynode' : 1,
              'reg_alpha' : 0,              # 알파, 람다 , L1 , L2 규제
            #   'reg_lamda' : 1,
              'random_state' : 220118 ,
            #   'verbose' : 0,
              }

#2 모델
# model = RandomForestClassifier(random_state = 980909 )
model = XGBClassifier(random_state = 12 )
model.set_params( **parameters ,
                #  early_stopping_rounds = 10 ,
                eval_metric = 'logloss',
                            )

#3 컴파일, 훈련
model.fit(x_train , y_train )

#4 평가
acc = model.score(x_test,y_test)
print('acc',acc)
predict = model.predict(x_test)
print('accuracy' , accuracy_score(y_test,predict) )


# acc 0.6715151515151515
# accuracy 0.6715151515151515
# parameters = {'n_estimators' : 1000, 
#               'learning_rate' : 0.5,
#               'max_depth': 10,               # 트리 깊이
#               'gamma' : 0,
#               'min_child_weight' : 3,       # 드랍 아웃 개념
#               'subsample' : 0.4,
#               'colsample_bytree' : 0.8,
#               'colsample_bylevel' : 0.7,
#               'colsample_bynode' : 1,
#               'reg_alpha' : 0,              # 알파, 람다 , L1 , L2 규제
#             #   'reg_lamda' : 1,
#               'random_state' : 3377,
#             #   'verbose' : 0,
#               }


# acc 0.6703030303030303
# accuracy 0.6703030303030303

# RandomForestClassifier
# Default 값으로 돌릴 값 
# acc 0.6981818181818182
# accuracy 0.6981818181818182

# outliers 사용
# acc 0.7054545454545454
# accuracy 0.7054545454545454

