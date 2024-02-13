import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold
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

print(train_csv.isna().sum())
print(test_csv.isna().sum())

# 결측 x

# LabelEncoder

# print(np.unique(test_csv['SMOKE'],return_counts=True))
# (array(['no', 'yes'], dtype=object), array([20513,   245], dtype=int64))
# (array(['no', 'yes'], dtype=object), array([13660,   180], dtype=int64))

# print(np.unique(test_csv['SCC'],return_counts=True))
# (array(['no', 'yes'], dtype=object), array([20071,   687], dtype=int64))
# (array(['no', 'yes'], dtype=object), array([13376,   464], dtype=int64))

# print(np.unique(test_csv['NObeyesdad'],return_counts=True))
# (array(['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I',
#        'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I',
#        'Overweight_Level_II'], dtype=object), array([2523, 3082, 2910, 3248, 4046, 2427, 2522], dtype=int64))

# replace
# print(np.unique(test_csv['CAEC'],return_counts=True))
# (array(['Always', 'Frequently', 'Sometimes', 'no'], dtype=object), array([  478,  2472, 17529,   279], dtype=int64))
# (array(['Always', 'Frequently', 'Sometimes', 'no'], dtype=object), array([  359,  1617, 11689,   175], dtype=int64))

# print(np.unique(test_csv['CALC'],return_counts=True))
# (array(['Frequently', 'Sometimes', 'no'], dtype=object), array([  529, 15066,  5163], dtype=int64))
# (array(['Always', 'Frequently', 'Sometimes', 'no'], dtype=object), array([   2,  346, 9979, 3513], dtype=int64))

# print(np.unique(test_csv['MTRANS'],return_counts=True))
# (array(['Automobile', 'Bike', 'Motorbike', 'Public_Transportation','Walking'], dtype=object),
#  array([        3534,     32,          38,                   16687,      467], dtype=int64))
# (array(['Automobile', 'Bike', 'Motorbike', 'Public_Transportation','Walking'], dtype=object), 
# array([         2405,     25,          19,                   11111,      280], dtype=int64))

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



x_train , x_test , y_train , y_test = train_test_split(x,y, random_state=123 , test_size=0.3 , shuffle=True , stratify=y )

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

es = EarlyStopping(monitor='val_loss', mode = 'min' , patience= 300 , restore_best_weights=True , verbose= 1 )


#2 모델구성
# model = RandomForestClassifier()
model = cb.CatBoostClassifier(eval_metric='acc', callback=[es] )       # ,auto_class_weights=True 
# model = xgb.XGBClassifier()
# model = lgbm.LGBMClassifier()

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits= 10 , shuffle=True , random_state= 730320 )
from sklearn.model_selection import cross_val_predict ,cross_val_score , GridSearchCV

parameters =[
    {'n_estimators' : [100,200] ,'max_depth':[6,10,12],'min_samples_leaf' : [3,10]},
    {'max_depth': [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3,5,7,10],'min_samples_split' : [2,3,5,10]},
    {'min_samples_split' : [2,3,5,10] },
    {'n_jobs' : [-1,2,4], 'min_samples_split' : [2,3,5,10]}
]

from scipy.stats import loguniform

catboost_grid = {
    'n_estimators': random.randint(100, 300),       # 랜덤으로 범위내 수를 뽑음
    'depth': random.randint(1, 5),                  # 랜덤으로 범위내 수를 뽑음
    'learning_rate': loguniform(1e-3, 0.1),         # 랜덤으로 범위내 수를 뽑음
    'min_child_samples': random.randint(10, 40),    # 랜덤으로 범위내 수를 뽑음
    'grow_policy': ['SymmetricTree', 'Lossguide', 'Depthwise']
}


# model = GridSearchCV(RandomForestClassifier() ,  parameters , cv=kfold,
#                      refit=True , verbose= 1 , n_jobs=-1 )


# model.randomized_search(catboost_grid,
#                             x_train, y_train,
#                             cv=kfold, n_iter=10 )

# random_search = RandomizedSearchCV(model, param_distributions=catboost_grid, n_iter=100, cv=5, random_state=730320 )
# random_search.fit(x_train,y_train)




# 최적의 하이퍼파라미터 출력
# print("Best hyperparameters:", random_search.best_params_)

# 최적의 모델 성능 출력
# print("Best score:", random_search.best_score_)

#3 훈련
model.fit(x_train,y_train)

#4 평가, 예측
# GridSearchCV 전용
y_predict = model.predict(x_test)
print('accuracy_score' , accuracy_score(y_test,y_predict))
print('='*100)
y_pred_best = model.best_estimator_.predict(x_test)
print('최적의 매겨번수:' , model.best_estimator_)
print('='*100)
print('최적의 튠 ACC:', accuracy_score(y_test,y_pred_best))

'''
score = cross_val_score(model,x_train , y_train , cv=kfold)
y_predict = cross_val_predict(model,x_test,y_test,cv=kfold)

print('acc',score)

acc= accuracy_score(y_test,y_predict)
print('ACC',acc)

'''
y_submit = model.predict(test_csv)

y_submit = le.inverse_transform(y_submit) 
submission_csv['NObeyesdad'] = y_submit

submission_csv.to_csv(path+'submission_0209.csv', index = False)


# RandomForestClassifier
# 0.8938

# Catboost
# 0.90679

# XGB
# 0.90462

# LGBM
# 0.90462

# DNN
# 0.86091
