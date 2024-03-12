import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split , StratifiedKFold , RandomizedSearchCV , GridSearchCV
import numpy as np
import dask
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import warnings
warnings.filterwarnings('ignore')
import time
from sklearn.preprocessing import PolynomialFeatures

#1 데이터
path = 'c:/_data/kaggle/fat//'

train_csv = pd.read_csv(path + 'train.csv',index_col=0)
test_csv = pd.read_csv(path + 'test.csv',index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

le = LabelEncoder()
le.fit(train_csv['Gender'])
train_csv['Gender'] = le.transform(train_csv['Gender'])
test_csv['Gender'] = le.transform(test_csv['Gender'])

le.fit(train_csv['family_history_with_overweight'])
train_csv['family_history_with_overweight'] = le.transform(train_csv['family_history_with_overweight'])
test_csv['family_history_with_overweight'] = le.transform(test_csv['family_history_with_overweight'])

le.fit(train_csv['FAVC'])
train_csv['FAVC'] = le.transform(train_csv['FAVC'])
test_csv['FAVC'] = le.transform(test_csv['FAVC'])

le.fit(train_csv['SMOKE'])
train_csv['SMOKE'] = le.transform(train_csv['SMOKE'])
test_csv['SMOKE'] = le.transform(test_csv['SMOKE'])

le.fit(train_csv['SCC'])
train_csv['SCC'] = le.transform(train_csv['SCC'])
test_csv['SCC'] = le.transform(test_csv['SCC'])

le.fit(train_csv['NObeyesdad'])
train_csv['NObeyesdad'] = le.transform(train_csv['NObeyesdad'])

train_csv['CAEC'] = train_csv['CAEC'].replace({'Always': 0 , 'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })
test_csv['CAEC'] = test_csv['CAEC'].replace({'Always': 0 , 'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })

train_csv['CALC'] = train_csv['CALC'].replace({'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })
test_csv['CALC'] = test_csv['CALC'].replace({'Always' : 2 , 'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })

train_csv['MTRANS'] = train_csv['MTRANS'].replace({'Automobile': 0 , 'Bike' : 1, 'Motorbike' : 2, 'Public_Transportation' : 3,'Walking' : 4})
test_csv['MTRANS'] = test_csv['MTRANS'].replace({'Automobile': 0 , 'Bike' : 1, 'Motorbike' : 2, 'Public_Transportation' : 3,'Walking' : 4})


train_csv = train_csv.drop(['SMOKE'],axis=1)
test_csv = test_csv.drop(['SMOKE'],axis=1)

x = train_csv.drop(['NObeyesdad'], axis= 1)
y = train_csv['NObeyesdad']

from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , RobustScaler

df = pd.DataFrame(x , columns = x.columns)
print(df)
df['target(Y)'] = y
print(df)

print('=================== 상관계수 히트맵 =====================')
print(df.corr())

pf = PolynomialFeatures(degree=2 , include_bias=False )
x1 = pf.fit_transform(x)

x_train , x_test , y_train , y_test = train_test_split(x1,y, random_state= 980909 , test_size=0.3 , shuffle=True , stratify=y )

# scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

x_train = np.array(x_train)
x_test = np.array(x_test)

#2 모델
from sklearn.ensemble import BaggingClassifier , RandomForestClassifier , VotingClassifier , StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

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

# xgb
# ACC 0.9059087989723827
# True
# ACC 0.8485870263326911
# False
# ACC 0.848747591522158


# soft
# ACC 0.9047848426461144
# hard
# ACC 0.901252408477842

# model.score :  0.8994861913937059
# accuracy :  0.8994861913937059

# PolynomialFeatures
# model.score :  0.8975594091201028
# accuracy :  0.8975594091201028

# best :  {'colsample_bytree': 0.7882110772719316, 'learning_rate': 0.0018521539284540906, 'max_bin': 446.0, 'max_depth': 3.0,
# 'min_child_samples': 69.0, 'min_child_weight': 33.0, 'num_leaves': 24.0, 'reg_alpha': 43.1284579284561, 
# 'reg_lambda': 0.811698575168277, 'subsample': 0.5875486510297028}
# 100 번 걸린시간 :  60.7943