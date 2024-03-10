from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn.preprocessing import PolynomialFeatures

#1 데이터

path = 'c:/_data/kaggle/bike//'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = train_csv.drop(['casual' , 'registered', 'count'], axis= 1 )        # [6493 rows x 8 columns] // drop을 줄 때 '를 따로 따로 줘야된다.
y = train_csv['count']

pf = PolynomialFeatures(degree=2, include_bias=False )
x1 = pf.fit_transform(x)

x_train , x_test , y_train , y_test = train_test_split(x1,y,test_size=0.3 , random_state= 6974 ) #7

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2 모델구성
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor , VotingRegressor , StackingRegressor
from sklearn.linear_model import LogisticRegressionCV , LogisticRegression
from sklearn.svm import LinearSVR
from catboost import CatBoostRegressor

xgb = XGBRegressor()
rf = RandomForestRegressor()
lr = LogisticRegressionCV()
svr = LinearSVR()

import warnings
warnings.filterwarnings('ignore')
import time
bayesian_params = {
    'learning_rate' : (0.001,1),
    'max_depth' : (3,10),
    'num_leaves' : (24,40),
    'min_child_samples' : (10,100),
    'min_child_weight' : (1,50),
    'subsample' : (0.5,1),
    'colsample_bytree' : (0.5,1),
    'max_bin' : (9,500),
    'reg_lambda' : (-0.001,10),
    'reg_alpha' : (0.01,50),
}


def xgb_hamsu( learning_rate, max_depth,num_leaves, min_child_samples, min_child_weight,subsample , colsample_bytree, max_bin,
              reg_lambda,reg_alpha) : 
    params = {
        'n_estimator' : 500,
        'learning_rate' :learning_rate,
        'max_depth' : int(round(max_depth)),        # 무조건 정수형
        'num_leaves' : int(round(num_leaves)),
        'min_child_samples' :int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1),0),          # 0~1 사이의 값만 뽑히게 해줌 최소를 1과 비교하고 최대를 0과 비교하여 나머지값이 오지 못하게 함
        'colsample_bytree' : colsample_bytree,
        'max_bin' : max(int(round(max_bin)),10),        # 무조건 10 이상
        'reg_lambda' :max(reg_lambda,0),                # 무조건 양수만
        'reg_alpha' : reg_alpha,
    }
    model = XGBRegressor(**params , n_jobs = -1 , )
    model.fit(x_train , y_train, eval_set = [(x_train,y_train), (x_test,y_test)],
            #   eval_metric = 'logloss',
              verbose = 0 , early_stopping_rounds = 50, )
    y_predict = model.predict(x_test)
    results = r2_score(y_test,y_predict)
    return results

from bayes_opt import BayesianOptimization
bay = BayesianOptimization(f = xgb_hamsu, pbounds=bayesian_params , random_state = 777 )

start = time.time()
n_iter = 100
bay.maximize(init_points=5 , n_iter=n_iter )
end = time.time()

print(bay.max)
print(n_iter, '번 걸린시간 : ', round(end - start,4) )


# model.score 0.3433454319382413
# True
# model.score -0.043975513382387144
# False
# model.score -0.3082794945444054


# model.score 0.36736569263577523

# model.score 0.34439630784791864

# PolynomialFeatures
# model.score 0.29891606848981866