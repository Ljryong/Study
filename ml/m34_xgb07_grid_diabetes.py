from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score
from keras.models import Sequential , load_model
from keras.layers import Dense
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)          # (442, 10) (442,)
print(datasets.feature_names)   #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# x = np.delete(x,(1,7),axis=1)

x = pd.DataFrame(x , columns =datasets.feature_names )
# x = x.drop(['sex', 's4'], axis = 1)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 151235 , shuffle= True )

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

n_splits = 3
from sklearn.model_selection import StratifiedKFold , RandomizedSearchCV , KFold
kfold = KFold(n_splits=n_splits ,random_state= 777 , shuffle=True )

#2 모델구성
from xgboost import XGBRegressor

model = RandomizedSearchCV(XGBRegressor() , parameters , cv=kfold , random_state= 777,
                           n_jobs=22 , n_iter= 10 )



#3 훈련
model.fit(x_train,y_train)

#4 평가, 예측
# GridSearchCV 전용
from sklearn.metrics import r2_score
score = model.score(x_test,y_test)
print('='*100)
print('매개변수 : ' , model.best_estimator_)
print('매개변수 : ' , model.best_params_)
print('점수 : ' ,score )



# 매개변수 :  XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=0.7, colsample_bynode=0.3, colsample_bytree=0.3,
#              device=None, early_stopping_rounds=None, enable_categorical=False,
#              eval_metric=None, feature_types=None, gamma=10, grow_policy=None,
#              importance_type=None, interaction_constraints=None,
#              learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#              max_cat_to_onehot=None, max_delta_step=None, max_depth=5,
#              max_leaves=None, min_child_weight=5, missing=nan,
#              monotone_constraints=None, multi_strategy=None, n_estimater=400,
#              n_estimators=None, n_jobs=None, num_parallel_tree=None, ...)
# 매개변수 :  {'subsample': 0.7, 'reg_lambda': 0.01, 'reg_alpha': 0.001, 'n_estimater': 400, 'min_child_weight': 5, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 10, 'colsample_bytree': 0.3, 'colsample_bynode': 0.3, 'colsample_bylevel': 0.7}
# 점수 :  0.4329768294404487