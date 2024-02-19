
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC


#1 
path = "c:/_data/dacon/wine//"

train_csv = pd.read_csv(path + "train.csv" , index_col= 0)      # index_col : 컬럼을 무시한다. //  index_col= 0 는 0번째 컬럼을 무시한다. 
test_csv = pd.read_csv(path + "test.csv" , index_col= 0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


# print(train_csv)        # [5497 rows x 13 columns]
# print(test_csv)         # [1000 rows x 12 columns]

# ######################## 사이킷런 문자데이터 수치화 ##################
# from sklearn.preprocessing import LabelEncoder      # 문자데이터를 알파벳 순서대로 수치화한다
# lab = LabelEncoder()
# lab.fit(train_csv)
# trainlab_csv = lab.transform(train_csv)
# print(trainlab_csv)


# #####################################################################

####### keras에 있는 데이터 수치화 방법 ##########
train_csv['type'] = train_csv['type'].replace({'white': 0, 'red':1})
test_csv['type'] = test_csv['type'].replace({'white': 0, 'red':1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']
# print(train_csv)
# print(y.shape)          # (5497,1)

print(np.unique(y))

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.3 , random_state= 971 , shuffle=True , stratify= y )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from xgboost import XGBClassifier


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

from sklearn.model_selection import StratifiedKFold

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits , random_state= 5 , shuffle= True)

#2
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
model = RandomizedSearchCV(XGBClassifier(tree_method = 'gpu_hist') , parameters , cv=kfold , random_state=777,
                           n_jobs= 22)

#3 훈련
model.fit(x_train,y_train-3)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
print(x.shape)



# ==================================================
# n_components =  1 result 0.4806060606060606
# ==================================================
# n_components =  2 result 0.5127272727272727
# ==================================================
# n_components =  3 result 0.5218181818181818
# ==================================================
# n_components =  4 result 0.5751515151515152
# ==================================================
# n_components =  5 result 0.616969696969697
# ==================================================
# n_components =  6 result 0.6163636363636363
# ==================================================
# n_components =  7 result 0.636969696969697
# ==================================================
# n_components =  8 result 0.636969696969697
# ==================================================
# n_components =  9 result 0.6442424242424243
# ==================================================
# n_components =  10 result 0.6393939393939394
# ==================================================
# n_components =  11 result 0.6575757575757576
# ==================================================
# n_components =  12 result 0.6460606060606061
# ==================================================


# model.score 0.6375757575757576
# (5497, 12)
# [0.31649163 0.52613257 0.6588525  0.73781223 0.79913175 0.84882972
#  0.89473275 0.93698174 0.96677118 0.98792024 0.99779749]

# model.score 0.6624242424242425
# (5497, 12)
# [0.84273258 0.95031682 0.9780272  0.99104808 0.99589734 1.        ]

# model.score 0.0036363636363636364 = 다시 만져보기