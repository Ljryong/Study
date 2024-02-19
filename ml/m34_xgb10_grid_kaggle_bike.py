from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR


#1 데이터

path = 'c:/_data/kaggle/bike//'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = train_csv.drop(['casual' , 'registered', 'count'], axis= 1 )        # [6493 rows x 8 columns] // drop을 줄 때 '를 따로 따로 줘야된다.
y = train_csv['count']


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state= 6974 ) #7

es = EarlyStopping(monitor = 'val_loss' , mode = 'min', patience = 10 , verbose= 1 ,restore_best_weights=True )

parameters  = {'n_estimater' : [100,200,600,1000], # 디폴트 100 / 1~inf / 정수
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

x_train, x_test, y_train , y_test = train_test_split(x,y,test_size = 0.3, random_state= 846 ) #45

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import RandomizedSearchCV , KFold
kfold = KFold(n_splits= 5 , random_state= 777 , shuffle=True )

#2 모델구성
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from xgboost import XGBRegressor
model = RandomizedSearchCV(XGBRegressor(tree_method = 'gpu_hist' , random_state = 40 ) , parameters , cv=kfold , random_state= 777 , n_iter=10 , n_jobs= 22 , )

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
print('매개변수',model.best_estimator_  )
print('매개변수',model.best_params_  )

# model.score 0.32470886705472224
# 매개변수 XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=0.7, colsample_bynode=0.3, colsample_bytree=0.3,
#              device=None, early_stopping_rounds=None, enable_categorical=False,
#              eval_metric=None, feature_types=None, gamma=10, grow_policy=None,
#              importance_type=None, interaction_constraints=None,
#              learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#              max_cat_to_onehot=None, max_delta_step=None, max_depth=5,
#              max_leaves=None, min_child_weight=5, missing=nan,
#              monotone_constraints=None, multi_strategy=None, n_estimater=400,
#              n_estimators=None, n_jobs=None, num_parallel_tree=None, ...)
# 매개변수 {'subsample': 0.7, 'reg_lambda': 0.01, 'reg_alpha': 0.001, 'n_estimater': 400, 'min_child_weight': 5,
#       'max_depth': 5, 'learning_rate': 0.1, 'gamma': 10, 'colsample_bytree': 0.3, 'colsample_bynode': 0.3, 'colsample_bylevel': 0.7}