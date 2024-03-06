import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.15 , random_state= 909 , stratify=y , shuffle = True )

scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators' : 1000, 
              'learning_rate' : 0.1,
              'max_depth': 10,               # 트리 깊이
              'gamma' : 0,
              'min_child_weight' : 3,       # 드랍 아웃 개념
              'subsample' : 0.4,
              'colsample_bytree' : 0.8,
              'colsample_bylevel' : 0.7,
              'colsample_bynode' : 1,
              'reg_alpha' : 0,              # 알파, 람다 , L1 , L2 규제
            #   'reg_lamda' : 1,
              'random_state' : 3377,
            #   'verbose' : 0,
             'tree_method' : 'gpu_hist',
              }

#2 모델
model = XGBClassifier(random_state = 980909 , )
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