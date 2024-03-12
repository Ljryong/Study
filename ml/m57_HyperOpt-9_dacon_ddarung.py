# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np          # 수치 계산이 빠름
import pandas as pd         # 수치 말고 다른 각종 계산들이 좋고 빠름
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.svm import LinearSVR
from sklearn.preprocessing import PolynomialFeatures


#1. 데이터

path = "c:/_data/dacon/ddarung//"
# print(path + "aaa_csv") = c:/_data/dacon/ddarung/aaa_csv


train_csv = pd.read_csv(path + "train.csv",index_col = 0) # index_col = 0 , 필요없는 열을 지울 때 사용한다 , index_col = 0 은 0번은 index야 라는 뜻
test_csv = pd.read_csv(path + "test.csv", index_col = 0)          # [715 rows x 10 columns] = [715,10] -- index_col = 0 사용하기 전 결과 값
submission_csv = pd.read_csv(path + "submission.csv", )   # 서브미션의 index_col을 사용하면 안됨 , 결과 틀에서 벗어날 수 있어서 index_col 을 사용하면 안됨

test_csv = test_csv.fillna(test_csv.mean())                    # 717 non-null     


##################### 결측치를 0으로 바꾸는 법#######################

train_csv = train_csv.fillna(0)

                                          

######### x 와 y 를 분리 ############
x = train_csv.drop(['count'],axis = 1)                # 'count'를 drop 해주세요 axis =1 에서 (count 행(axis = 1)을 drop 해주세요) // 원본을 건드리는 것이 아니라 이 함수만 해당
print(x)
y = train_csv['count']                                # count 만 가져오겠다
print(y)

pf = PolynomialFeatures(degree=2 , include_bias=False )
x1 = pf.fit_transform(x)

x_train, x_test, y_train , y_test = train_test_split(x1,y,test_size = 0.3, random_state= 846 ) #45

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import RandomizedSearchCV , KFold
kfold = KFold(n_splits= 5 , random_state= 777 , shuffle=True )

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




# model.score 0.7985347257265492

# True
# model.score 0.5851694624663715
# False
# model.score 0.43054496016231514

# model.score 0.7689441977396401

# model.score 0.7904146861441637

# best :  {'colsample_bytree': 0.99635137435327, 'learning_rate': 0.0014989643637105397, 'max_bin': 280.0, 'max_depth': 9.0,
# 'min_child_samples': 42.0, 'min_child_weight': 47.0, 'num_leaves': 28.0, 'reg_alpha': 28.271916625269114, 
# 'reg_lambda': 2.2337563125222974, 'subsample': 0.9639773645054783}
# 100 번 걸린시간 :  5.8293