from sklearn.datasets import fetch_california_housing , _california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import time                                 # 시간에 대한 정보를 가져온다
from sklearn.svm import LinearSVR
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

#1
datasets = fetch_california_housing()
# print(datasets.items())
x = datasets.data
y = datasets.target

pf = PolynomialFeatures(degree=2, include_bias=False)
x1 = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size = 0.3, random_state = 59 )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor , VotingRegressor , StackingRegressor
from sklearn.linear_model import LogisticRegressionCV , LogisticRegression
from sklearn.svm import LinearSVR
from catboost import CatBoostRegressor

xgb = XGBRegressor()
rf = RandomForestRegressor()
lr = LogisticRegressionCV()
svr = LinearSVR()

import time
import warnings
warnings.filterwarnings('ignore')
from hyperopt import hp , fmin , Trials, tpe

search_space = {
    'learning_rate' : hp.uniform('learning_rate',0.001,0.1),
    'max_depth' : hp.quniform('max_depth',3,10,1),
    'num_leaves' : hp.quniform('num_leaves',24,40,1),
    'min_child_samples' : hp.quniform('min_child_samples',10,100,1),
    'min_child_weight' : hp.quniform('min_child_weight',1,50,1),
    'subsample' : hp.uniform('subsample',0.5,1),
    'colsample_bytree' : hp.uniform('colsample_bytree',0.5,1),
    'max_bin' : hp.quniform('max_bin',9,500,1),
    'reg_lambda' : hp.uniform('reg_lambda',-0.001,10),
    'reg_alpha' : hp.uniform('reg_alpha',0.01,50),
}


def xgb_hamsu(search_space) : 
    params = {
        'n_estimator' : 100,
        'learning_rate' : search_space['learning_rate'],
        'max_depth' : int(search_space['max_depth']),        # 무조건 정수형
        'num_leaves' : int(search_space['num_leaves']),
        'min_child_samples' : int(search_space['min_child_samples']),
        'min_child_weight' : int(search_space['min_child_weight']),
        'subsample' : max(min(search_space['subsample'],1),0),          # 0~1 사이의 값만 뽑히게 해줌 최소를 1과 비교하고 최대를 0과 비교하여 나머지값이 오지 못하게 함
        'colsample_bytree' : search_space['colsample_bytree'],
        'max_bin' : max(int(search_space['max_bin']),10),        # 무조건 10 이상
        'reg_lambda' :max(search_space['reg_lambda'],0),                # 무조건 양수만
        'reg_alpha' : search_space['reg_alpha'],
    }
    model = XGBRegressor(**params , n_jobs = -1 , )
    model.fit(x_train , y_train, eval_set = [(x_train,y_train), (x_test,y_test)],
            #   eval_metric = 'mlogloss',
              verbose = 0 , early_stopping_rounds = 50, )
    y_predict = model.predict(x_test)
    results = r2_score(y_test,y_predict)
    return results

trial_val = Trials()

n_iter = 100

start = time.time()
best = fmin(
    fn= xgb_hamsu,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trial_val,
    rstate=np.random.default_rng(seed=10)
)

end = time.time()

print('best : ',best)
print(n_iter, '번 걸린시간 : ', round(end - start,4) )


# model.score 0.8401449795919554
# True
# model.score 0.6220342782718715
# False
# model.score 0.6127496034406276


# model.score 0.8102075541825241

# model.score 0.8402800944673787

# PolynomialFeatures
# model.score 0.8061424615032632

# best :  {'colsample_bytree': 0.8647398255924487, 'learning_rate': 0.0014989643637105397, 'max_bin': 145.0, 'max_depth': 9.0,
# 'min_child_samples': 62.0, 'min_child_weight': 38.0, 'num_leaves': 24.0, 'reg_alpha': 27.680948409372547,
# 'reg_lambda': 2.2337563125222974, 'subsample': 0.9639773645054783}
# 100 번 걸린시간 :  15.5221