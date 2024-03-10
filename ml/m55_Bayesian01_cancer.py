import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler ,StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score , r2_score , mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier , VotingClassifier , StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import random as rn
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import time

# 1.데이터
x, y =load_breast_cancer(return_X_y=True)

pf = PolynomialFeatures( degree = 2 , include_bias=False )
x_poly = pf.fit_transform(x)

x_train , x_test , y_train , y_test = train_test_split(x_poly,y,random_state=777 , train_size=0.8 ,stratify=y )

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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
    model = XGBClassifier(**params , n_jobs = -1 , )
    model.fit(x_train , y_train, eval_set = [(x_train,y_train), (x_test,y_test)], eval_metric = 'logloss',verbose = 0 , early_stopping_rounds = 50, )
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test,y_predict)
    return results

bay = BayesianOptimization(f = xgb_hamsu, pbounds=bayesian_params , random_state = 777 )

start = time.time()
n_iter = 100
bay.maximize(init_points=5 , n_iter=n_iter )
end = time.time()

print(bay.max)
print(n_iter, '번 걸린시간 : ', round(end - start,4) )
# {'target': 0.9912280701754386, 'params': {'colsample_bytree': 0.9414546180348633, 'learning_rate': 0.434508080080861, 'max_bin': 35.18613252492072, 'max_depth': 3.5129804658661454, 'min_child_samples': 47.59498868412147, 'min_child_weight': 4.176422496862631, 'num_leaves': 34.37311983841647, 'reg_alpha': 0.7364028450117082, 'reg_lambda': 9.225949121620184, 'subsample': 0.566394725317649}}
# 500 번 걸린시간 :  246.2861