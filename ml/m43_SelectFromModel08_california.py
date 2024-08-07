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
from xgboost import XGBRegressor
model = XGBRegressor(tree_method = 'gpu_hist') 

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
print(x.shape)

from sklearn.metrics import accuracy_score

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
    
    select_model = XGBRegressor()
    select_model.set_params(early_stopping_rounds = 10 , **parameters ,
                            # eval_metric = 'logloss'
                            )
    
    select_model.fit(select_x_train,y_train  , eval_set = [(select_x_train , y_train  ),(select_x_test,y_test  )], verbose = 0 ) 
    
    
    select_y_predict = select_model.predict(select_x_test)
    score = r2_score(y_test , select_y_predict)
    
    print("Thredsholds=%.3f, n=%d, ACC: %.2f%%" %(i, select_x_train.shape[1], score*100))



# Thredsholds=0.025, n=8, ACC: 81.90%
# Thredsholds=0.025, n=7, ACC: 82.13%
# Thredsholds=0.045, n=6, ACC: 83.02%
# Thredsholds=0.068, n=5, ACC: 82.70%
# Thredsholds=0.099, n=4, ACC: 81.50%
# Thredsholds=0.106, n=3, ACC: 71.88%
# Thredsholds=0.154, n=2, ACC: 59.25%
# Thredsholds=0.479, n=1, ACC: 50.42%