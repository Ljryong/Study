import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split , StratifiedKFold , RandomizedSearchCV , GridSearchCV
import numpy as np
import dask
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import warnings
warnings.filterwarnings('ignore')
import time

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


train_csv = train_csv.drop(['SMOKE'],axis=1)
test_csv = test_csv.drop(['SMOKE'],axis=1)

x = train_csv.drop(['NObeyesdad'], axis= 1)
y = train_csv['NObeyesdad']

from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , RobustScaler

df = pd.DataFrame(x , columns = x.columns)
print(df)
df['target(Y)'] = y
print(df)

print('=================== 상관계수 히트맵 =====================')
print(df.corr())

x_train , x_test , y_train , y_test = train_test_split(x,y, random_state= 980909 , test_size=0.3 , shuffle=True , stratify=y )

# scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

x_train = np.array(x_train)
x_test = np.array(x_test)

kfold = StratifiedKFold(n_splits= 3 , shuffle=True , random_state= 730501 )

import random
xgb_grid = {'n_estimater' : [200,400,500,1000], # 디폴트 100 / 1~inf / 정수
'learning_rate' : [0.1,0.2,0.01,0.001], # 디폴트 0.3 / 0~1 / eta 제일 중요  
# learning_rate(훈련율) : 작을수록 디테일하게 보고 크면 클수록 듬성듬성 본다. batch_size랑 비슷한 느낌
#                        하지만 너무 작으면 오래 걸림 데이터의 따라 잘 조절 해야된다
'max_depth' : [2,3,4,5], # 디폴트 6 / 0~inf / 정수    # tree의 깊이를 나타냄
'gamma' : [0,1,2,3,4,5,7,10,100], # 디폴트 0 / 0~inf 
'min_child_weight' : [0,0.01,0.001,0.1,0.5,1,5,10,100], # 디폴트 1 / 0~inf 
'subsample' : [0,0.1,0.2,0.3,0.5,0.7,1], # 디폴트 1 / 0~1
'colsample_bytree' : [0,0.1,0.2,0.3,0.5,0.7,1], # 디폴트 1 / 0~1 
'colsample_bylevel' : [0,0.1,0.2,0.3,0.5,0.7,1], # 디폴트 1 / 0~1
'colsample_bynode' : [0,0.1,0.2,0.3,0.5,0.7,1], # 디폴트 1 / 0~1
'reg_alpha' : [0,0.1,0.01,0.001,1,2,10], # 디폴트 0 / 0~inf / L1 절대값 가중치 규제 / alpha / 중요
'reg_lambda' : [0,0.1,0.01,0.001,1,2,10],# 디폴트 1 / 0~inf / L2 제곱 가중치 규제 / lambda / 중요
}


model = xgb.XGBClassifier(random_state= 220118 ,tree_method='gpu_hist')


# xgb gpu 사용방법
# tree_method = 'gpu_hist'

#3 훈련
model.fit(x_train,y_train)

#4 평가, 예측
# GridSearchCV 전용
from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('='*100)
acc= accuracy_score(y_test,y_predict)
print('ACC',acc)
y_submit = model.predict(test_csv)

y_submit = le.inverse_transform(y_submit) 
submission_csv['NObeyesdad'] = y_submit

submission_csv.to_csv(path+'submission_xgb.csv', index = False)


# 초기 특성 중요도
import warnings
warnings.filterwarnings('ignore')
feature_importances = model.feature_importances_
sort= np.argsort(feature_importances)               # argsort 열의 번호로 반환해줌
print(sort)

removed_features = 0

# 각 반복에서 피처를 추가로 제거하면서 성능 평가
for i in range(len(model.feature_importances_) - 1):
    remove = sort[:i+1]  # 추가로 제거할 피처의 인덱스
    
    print(f"Removing features at indices: {remove}")
    
    # 해당 특성 제거
    x_train_removed = np.delete(x_train, remove, axis=1)
    x_test_removed = np.delete(x_test, remove, axis=1)

    # 모델 재구성 및 훈련
    model.fit(x_train_removed, y_train, eval_set=[(x_train_removed, y_train), (x_test_removed, y_test)],
              verbose=0, eval_metric='mlogloss', early_stopping_rounds=10)
    
    # 모델 평가
    acc = model.score(x_test_removed, y_test)
    print('Accuracy after removing features:', acc)
    
    # 제거된 피처의 개수를 누적
    removed_features += 1
    print(f"Total number of removed features: {removed_features}\n")

# ACC 0.9059087989723827
# [11 14 12  8  1  4 10  7  9 13  5  2  6  3  0]
# Removing features at indices: [11]
# Accuracy after removing features: 0.9041425818882466
# Total number of removed features: 1

# Removing features at indices: [11 14]
# Accuracy after removing features: 0.9043031470777135
# Total number of removed features: 2

# Removing features at indices: [11 14 12]
# Accuracy after removing features: 0.901252408477842
# Total number of removed features: 3

# Removing features at indices: [11 14 12  8]
# Accuracy after removing features: 0.9022157996146436
# Total number of removed features: 4

# Removing features at indices: [11 14 12  8  1]
# Accuracy after removing features: 0.8972382787411689
# Total number of removed features: 5

# Removing features at indices: [11 14 12  8  1  4]
# Accuracy after removing features: 0.8953114964675658
# Total number of removed features: 6

# Removing features at indices: [11 14 12  8  1  4 10]
# Accuracy after removing features: 0.8959537572254336
# Total number of removed features: 7

# Removing features at indices: [11 14 12  8  1  4 10  7]
# Accuracy after removing features: 0.8946692357096981
# Total number of removed features: 8

# Removing features at indices: [11 14 12  8  1  4 10  7  9]
# Accuracy after removing features: 0.8880860629415542
# Total number of removed features: 9

# Removing features at indices: [11 14 12  8  1  4 10  7  9 13]
# Accuracy after removing features: 0.8856775850995504
# Total number of removed features: 10

# Removing features at indices: [11 14 12  8  1  4 10  7  9 13  5]
# Accuracy after removing features: 0.8840719332048812
# Total number of removed features: 11

# Removing features at indices: [11 14 12  8  1  4 10  7  9 13  5  2]
# Accuracy after removing features: 0.8198458574181118
# Total number of removed features: 12

# Removing features at indices: [11 14 12  8  1  4 10  7  9 13  5  2  6]
# Accuracy after removing features: 0.8068400770712909
# Total number of removed features: 13

# Removing features at indices: [11 14 12  8  1  4 10  7  9 13  5  2  6  3]
# Accuracy after removing features: 0.3503532434168272
# Total number of removed features: 14