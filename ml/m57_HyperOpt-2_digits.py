from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense 
import time
from sklearn.model_selection import train_test_split , RandomizedSearchCV , GridSearchCV , StratifiedKFold , cross_val_predict , cross_val_score
from sklearn.ensemble import RandomForestClassifier , BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

datasets = load_digits()
x = datasets.data
y = datasets.target

pf = PolynomialFeatures( degree= 2 , include_bias=False )
x_poly = pf.fit_transform(x)

x_train, x_test , y_train , y_test = train_test_split(x_poly,y,test_size= 0.3 , random_state= 2222 , stratify=y , shuffle=True)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor , VotingClassifier , StackingClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from catboost import CatBoostClassifier
xgb = XGBClassifier()

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
# model.score 0.9518518518518518
# True
# model.score 0.9703703703703703
# False
# model.score 0.9703703703703703

# soft
# model.score 0.9777777777777777
# hard
# model.score 0.9777777777777777

# model.score :  0.9833333333333333
# accuracy :  0.9833333333333333

# PolynomialFeatures
# model.score :  0.9722222222222222
# accuracy :  0.9722222222222222

# best :  {'colsample_bytree': 0.8451637475834627, 'learning_rate': 0.04535632530367916, 'max_bin': 308.0, 
# 'max_depth': 10.0, 'min_child_samples': 70.0, 'min_child_weight': 43.0, 'num_leaves': 26.0, 'reg_alpha': 43.76153800972882, 
# 'reg_lambda': 3.468758061173138, 'subsample': 0.500230349376612}
# 100 번 걸린시간 :  172.6287