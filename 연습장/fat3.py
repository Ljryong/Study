import pandas as pd
from sklearn.model_selection import train_test_split, KFold  , StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import catboost as cb
import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier
from keras.callbacks import EarlyStopping
import random

#1 데이터
path = 'c:/_data/kaggle/fat//'

train_csv = pd.read_csv(path + 'train.csv',index_col=0)
test_csv = pd.read_csv(path + 'test.csv',index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

le = LabelEncoder()
le.fit(train_csv['Gender'])
train_csv['Gender'] = le.transform(train_csv['Gender'])
test_csv['Gender'] = le.transform(test_csv['Gender'])

le.fit(train_csv['family_history_with_overweight'])
train_csv['family_history_with_overweight'] = le.transform(train_csv['family_history_with_overweight'])
test_csv['family_history_with_overweight'] = le.transform(test_csv['family_history_with_overweight'])

le.fit(train_csv['FAVC'])
train_csv['FAVC'] = le.transform(train_csv['FAVC'])
test_csv['FAVC'] = le.transform(test_csv['FAVC'])

le.fit(train_csv['SMOKE'])
train_csv['SMOKE'] = le.transform(train_csv['SMOKE'])
test_csv['SMOKE'] = le.transform(test_csv['SMOKE'])

le.fit(train_csv['SCC'])
train_csv['SCC'] = le.transform(train_csv['SCC'])
test_csv['SCC'] = le.transform(test_csv['SCC'])

le.fit(train_csv['NObeyesdad'])
train_csv['NObeyesdad'] = le.transform(train_csv['NObeyesdad'])

train_csv['CAEC'] = train_csv['CAEC'].replace({'Always': 0 , 'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })
test_csv['CAEC'] = test_csv['CAEC'].replace({'Always': 0 , 'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })

train_csv['CALC'] = train_csv['CALC'].replace({'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })
test_csv['CALC'] = test_csv['CALC'].replace({'Always' : 2 , 'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })

train_csv['MTRANS'] = train_csv['MTRANS'].replace({'Automobile': 0 , 'Bike' : 1, 'Motorbike' : 2, 'Public_Transportation' : 3,'Walking' : 4})
test_csv['MTRANS'] = test_csv['MTRANS'].replace({'Automobile': 0 , 'Bike' : 1, 'Motorbike' : 2, 'Public_Transportation' : 3,'Walking' : 4})

x = train_csv.drop(['NObeyesdad'], axis= 1)
y = train_csv['NObeyesdad']

from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , RobustScaler


x_train , x_test , y_train , y_test = train_test_split(x,y, random_state= 2220 , test_size=0.3 , shuffle=True , stratify=y )

scaler = StandardScaler()
# scaler = MinMaxScaler()
# 스케일러 상관 없음 점수가 똑같음
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

es = EarlyStopping(monitor='val_loss', mode = 'min' , patience= 300 , restore_best_weights=True , verbose= 1 )

kfold = StratifiedKFold(n_splits= 10 , shuffle=True , random_state= 2220 )

#2 모델

# model = lgbm.LGBMClassifier()
model = xgb.XGBClassifier()
# model = cb.CatBoostClassifier()

#3 훈련
from sklearn.model_selection import cross_val_predict
model.fit(x_train,y_train)

#4 평가
acc = cross_val_predict(model,x_test,y_test , verbose=1 , cv=kfold )

print('Acc',acc)

y_submit = model.predict(test_csv)

y_submit = le.inverse_transform(y_submit) 
submission_csv['NObeyesdad'] = y_submit

submission_csv.to_csv(path+'submission_0210.csv', index = False)




