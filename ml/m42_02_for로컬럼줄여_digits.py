from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense 
import time
from sklearn.model_selection import train_test_split , RandomizedSearchCV , GridSearchCV , StratifiedKFold , cross_val_predict , cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# import numpy as np

datasets = load_digits()
x = datasets.data
y = datasets.target

x_train, x_test , y_train , y_test = train_test_split(x,y,test_size= 0.3 , random_state= 2222 , stratify=y , shuffle=True)

  
'''parameters  = {'n_estimater' : [400,500,1000], # 디폴트 100 / 1~inf / 정수
'learning_rate' : [0.1,0.2,0.01,0.001], # 디폴트 0.3 / 0~1 / eta 제일 중요  
# learning_rate(훈련율) : 작을수록 디테일하게 보고 크면 클수록 듬성듬성 본다. batch_size랑 비슷한 느낌
#                        하지만 너무 작으면 오래 걸림 데이터의 따라 잘 조절 해야된다
'max_depth' : [2,3,4,5], # 디폴트 6 / 0~inf / 정수    # tree의 깊이를 나타냄
'gamma' : [1,2,3,4,5,7,10,100], # 디폴트 0 / 0~inf 
'min_child_weight' : [0,0.5,1,5,10,100], # 디폴트 1 / 0~inf 
'subsample' : [0,0.1,0.2,0.3,0.5,0.7,1], # 디폴트 1 / 0~1
'colsample_bytree' : [0.1,0.2,0.3,0.5,0.7,1], # 디폴트 1 / 0~1 
'colsample_bylevel' : [0,0.1,0.2,0.3,0.5,0.7,1], # 디폴트 1 / 0~1
'colsample_bynode' : [0,0.1,0.2,0.3,0.5,0.7,1], # 디폴트 1 / 0~1
'reg_alpha' : [0,0.1,0.01,0.001,1,2,10], # 디폴트 0 / 0~inf / L1 절대값 가중치 규제 / alpha / 중요
'reg_lambda' : [0,0.1,0.01,0.001,1,2,10],# 디폴트 1 / 0~inf / L2 제곱 가중치 규제 / lambda / 중요
}'''

kfold = StratifiedKFold(n_splits= 3 , shuffle=True , random_state= 2222 )

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
from xgboost import XGBClassifier, XGBRFClassifier

model = XGBClassifier() 
model.set_params( random_state = 777)


#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
# print('매개변수', model.best_estimator_ )
# print('매개변수', model.best_params_ )

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



# model.score 0.9518518518518518
# (1797, 64)
# [0.28060193 0.46816127 0.63866271 0.75642415 0.84340146 0.90838518
#  0.95120279 0.98085574 1.        ]



# model.score 0.9555555555555556
# 매개변수 XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=0.2, colsample_bynode=0.7, colsample_bytree=1,
#               device=None, early_stopping_rounds=None, enable_categorical=False,
#               eval_metric=None, feature_types=None, gamma=2, grow_policy=None,
#               importance_type=None, interaction_constraints=None,
#               learning_rate=0.001, max_bin=None, max_cat_threshold=None,
#               max_cat_to_onehot=None, max_delta_step=None, max_depth=4,
#               max_leaves=None, min_child_weight=0.5, missing=nan,
#               monotone_constraints=None, multi_strategy=None, n_estimater=500,
#               n_estimators=None, n_jobs=None, num_parallel_tree=None, ...)
# 매개변수 {'subsample': 0.5, 'reg_lambda': 0, 'reg_alpha': 2, 'n_estimater': 500, 'min_child_weight': 0.5, 'max_depth': 4, 
#       'learning_rate': 0.001, 'gamma': 2, 'colsample_bytree': 1, 'colsample_bynode': 0.7, 'colsample_bylevel': 0.2}


# model.score 0.9611111111111111
# Removing features at indices: [0]
# Accuracy after removing features: 0.9611111111111111
# Total number of removed features: 1~13 이 최고점 결국 해도 안해도 똑같음