from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold , StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import time

#1
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
y = y - 1

pf = PolynomialFeatures(degree = 2 , include_bias=False )
x_poly = pf.fit_transform(x)


print(x.shape,y.shape)      # (581012, 54) (581012,)
print(pd.value_counts(y))   # 2    283301 , 1    211840 , 3     35754 , 7     20510 , 6     17367 , 5      9493 , 4      2747   (n,7)

x_train , x_test , y_train , y_test = train_test_split(x_poly,y ,test_size=0.3 , random_state= 2222 ,shuffle=True, stratify=y ) # 0

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier , XGBRegressor
from sklearn.ensemble import RandomForestClassifier , VotingClassifier , StackingClassifier
from catboost import CatBoostClassifier

xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

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
    model = XGBClassifier(**params , n_jobs = -1 , )
    model.fit(x_train , y_train, eval_set = [(x_train,y_train), (x_test,y_test)], eval_metric = 'mlogloss',verbose = 0 , early_stopping_rounds = 50, )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test,y_predict)
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

# 원래 값
# model.score 0.10956145584725537

# False
# model.score 0.7258640077106664

# True
# model.score 0.7258295850927116

# soft
# model.score 0.899795759133468
# hard
# model.score 0.884414586010648

# PolynomialFeatures
# model.score :  0.9539654855883973
# accuracy :  0.9539654855883973