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

parameters = {'n_estimators' : 1000, 
              'learning_rate' : 0.1,
              'max_depth': 3,               # 트리 깊이
              'gamma' : 0,
              'min_child_weight' : 0,       # 드랍 아웃 개념
              'subsample' : 0.4,
              'colsample_bytree' : 0.8,
              'colsample_bylevel' : 0.7,
              'colsample_bynode' : 1,
              'reg_alpha' : 0,              # 알파, 람다 , L1 , L2 규제
              'reg_lamda' : 1,
              'random_state' : 3377,
              'verbose' : 0,
              }

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
from sklearn.feature_selection import SelectFromModel
warnings.filterwarnings('ignore')
thresholds = np.sort(model.feature_importances_)
print(thresholds)

for i in thresholds:                                                    # 제일 작은것들을 먼저 없애줌
    # i 보다 크거나 같은 것만 남음 
    selection =  SelectFromModel(model, threshold=i ,prefit=False)        # selectionws은 인스턴스(변수)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(i ,'\t변형된 x_train',select_x_train.shape, i ,'변형된 x_test',select_x_test.shape)
    
    select_model = XGBClassifier()
    select_model.set_params(early_stopping_rounds = 10 , **parameters ,
                            # eval_metric = 'logloss'
                            )
    
    select_model.fit(select_x_train,y_train , eval_set = [(select_x_train , y_train),(select_x_test,y_test)], verbose = 0 ) 
    
    
    select_y_predict = select_model.predict(select_x_test)
    score = accuracy_score(y_test , select_y_predict)
    
    print("Thredsholds=%.3f, n=%d, ACC: %.2f%%" %(i, select_x_train.shape[1], score*100))


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

# Thredsholds=0.000, n=64, ACC: 97.78%
# Thredsholds=0.000, n=64, ACC: 97.78%
# Thredsholds=0.000, n=64, ACC: 97.78%
# Thredsholds=0.000, n=64, ACC: 97.78%
# Thredsholds=0.000, n=64, ACC: 97.78%
# Thredsholds=0.000, n=64, ACC: 97.78%
# Thredsholds=0.000, n=64, ACC: 97.78%
# Thredsholds=0.000, n=64, ACC: 97.78%
# Thredsholds=0.000, n=64, ACC: 97.78%
# Thredsholds=0.000, n=64, ACC: 97.78%
# Thredsholds=0.000, n=64, ACC: 97.78%
# Thredsholds=0.000, n=64, ACC: 97.78%
# Thredsholds=0.000, n=64, ACC: 97.78%
# Thredsholds=0.002, n=51, ACC: 97.78%
# Thredsholds=0.003, n=50, ACC: 97.59%
# Thredsholds=0.004, n=49, ACC: 97.41%
# Thredsholds=0.005, n=48, ACC: 97.78%
# Thredsholds=0.005, n=47, ACC: 97.96%
# Thredsholds=0.005, n=46, ACC: 97.22%
# Thredsholds=0.006, n=45, ACC: 96.85%
# Thredsholds=0.006, n=44, ACC: 97.41%
# Thredsholds=0.006, n=43, ACC: 97.59%
# Thredsholds=0.006, n=42, ACC: 97.41%
# Thredsholds=0.006, n=41, ACC: 97.78%
# Thredsholds=0.006, n=40, ACC: 97.78%
# Thredsholds=0.006, n=39, ACC: 97.41%
# Thredsholds=0.007, n=38, ACC: 97.41%
# Thredsholds=0.008, n=37, ACC: 97.04%
# Thredsholds=0.008, n=36, ACC: 97.04%
# Thredsholds=0.008, n=35, ACC: 97.41%
# Thredsholds=0.009, n=34, ACC: 97.22%
# Thredsholds=0.009, n=33, ACC: 97.04%
# Thredsholds=0.009, n=32, ACC: 97.22%
# Thredsholds=0.009, n=31, ACC: 97.04%
# Thredsholds=0.009, n=30, ACC: 96.48%
# Thredsholds=0.010, n=29, ACC: 97.04%
# Thredsholds=0.010, n=28, ACC: 97.04%
# Thredsholds=0.010, n=27, ACC: 97.04%
# Thredsholds=0.011, n=26, ACC: 96.48%
# Thredsholds=0.011, n=25, ACC: 96.85%
# Thredsholds=0.011, n=24, ACC: 96.67%
# Thredsholds=0.011, n=23, ACC: 96.30%
# Thredsholds=0.013, n=22, ACC: 95.19%
# Thredsholds=0.013, n=21, ACC: 95.56%
# Thredsholds=0.016, n=20, ACC: 94.63%
# Thredsholds=0.020, n=19, ACC: 93.70%
# Thredsholds=0.021, n=18, ACC: 93.89%
# Thredsholds=0.024, n=17, ACC: 93.52%
# Thredsholds=0.027, n=16, ACC: 93.52%
# Thredsholds=0.027, n=15, ACC: 92.96%
# Thredsholds=0.029, n=14, ACC: 92.04%
# Thredsholds=0.030, n=13, ACC: 91.85%
# Thredsholds=0.031, n=12, ACC: 89.63%
# Thredsholds=0.035, n=11, ACC: 88.52%
# Thredsholds=0.035, n=10, ACC: 85.74%
# Thredsholds=0.037, n=9, ACC: 82.96%
# Thredsholds=0.038, n=8, ACC: 78.70%
# Thredsholds=0.038, n=7, ACC: 74.07%
# Thredsholds=0.041, n=6, ACC: 73.52%
# Thredsholds=0.049, n=5, ACC: 69.07%
# Thredsholds=0.051, n=4, ACC: 57.22%
# Thredsholds=0.053, n=3, ACC: 51.85%
# Thredsholds=0.058, n=2, ACC: 34.07%
# Thredsholds=0.098, n=1, ACC: 25.74%