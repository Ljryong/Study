
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures

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
y = train_csv['quality'] - 3
# print(train_csv)
# print(y.shape)          # (5497,1)

pf = PolynomialFeatures(degree=2 , include_bias=False )
x1 = pf.fit_transform(x)

x_train , x_test , y_train , y_test = train_test_split(x1,y, test_size=0.3 , random_state= 971 , shuffle=True , stratify= y )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

es = EarlyStopping(monitor='val_loss' , mode = 'min', verbose=1, patience= 100 , restore_best_weights=True )

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor , BaggingClassifier , VotingClassifier , StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

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

# model.score 0.6557575757575758
# True
# model.score 0.5436363636363636
# False
# model.score 0.5430303030303031

# soft
# model.score 0.6533333333333333
# hard
# model.score 0.6454545454545455


# model.score :  0.6533333333333333
# accuracy :  0.6533333333333333

# PolynomialFeatures
# model.score :  0.6533333333333333
# accuracy :  0.6533333333333333

# best :  {'colsample_bytree': 0.8020212110221104, 'learning_rate': 0.0010210283435134887, 'max_bin': 268.0, 'max_depth': 10.0, 
# 'min_child_samples': 70.0, 'min_child_weight': 50.0, 'num_leaves': 27.0, 'reg_alpha': 49.91321579818407, 'reg_lambda': 2.451929692695452,
# 'subsample': 0.7060597598241505}
# 100 번 걸린시간 :  26.4073