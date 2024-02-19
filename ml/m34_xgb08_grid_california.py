from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import time                                 # 시간에 대한 정보를 가져온다
from sklearn.svm import LinearSVR
import pandas as pd

#1
datasets = fetch_california_housing()
print(datasets.items())
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 59 )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

parameters  = {'n_estimater' : [200,400,500,1000], # 디폴트 100 / 1~inf / 정수
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

from sklearn.model_selection import RandomizedSearchCV , StratifiedKFold , KFold
kfold = KFold(n_splits= 5 , random_state= 777 , shuffle=True)


#2
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from xgboost import XGBRegressor
model = RandomizedSearchCV(XGBRegressor(tree_method = 'gpu_hist') , parameters , cv = kfold , random_state= 777 , n_jobs= 22 , n_iter= 10 )

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
print(x.shape)


# n_components =  1 result -0.28710211497111615
# ==================================================
# n_components =  2 result 0.19122223901649837
# ==================================================
# n_components =  3 result 0.3746994061251102
# ==================================================
# n_components =  4 result 0.5812858524375956
# ==================================================
# n_components =  5 result 0.6081246215384929
# ==================================================
# n_components =  6 result 0.6600792725331971
# ==================================================
# n_components =  7 result 0.7262655783180297
# ==================================================
# n_components =  8 result 0.7407074320320721
# ==================================================


# model.score 0.7399965968397997
# (20640, 8)
# [0.25324198 0.48881928 0.64784913 0.77668622 0.90217681 0.9843977
#  0.99428932 1.        ]



