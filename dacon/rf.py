import pandas as pd
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import tensorflow as tf
import random as rn

SEED1 = 42
SEED2 = 42
SEED3 = 42

tf.random.set_seed(SEED1)  
np.random.seed(SEED2)
rn.seed(SEED3)

path = 'C:\_data\dacon\RF\\'
data = pd.read_csv(path + 'train.csv')
submit = pd.read_csv(path + 'sample_submission.csv')

# person_id 컬럼 제거
X_train = data.drop(['person_id', 'login'], axis=1)
y_train = data['login']

# GridSearchCV를 위한 하이퍼파라미터 설정
param_search_space = {
    'n_estimators': [100,300,500,700, 1000],  
    'criterion': ['gini', 'entropy'],  
    'max_depth': [ 5, 25, 55, 68 ],
    'min_samples_split': [ 0.01,0.05,0.1,0.5,0.7,2 ],
    'min_samples_leaf':[ 0.01,0.04,0.08,0.1,0.3,0.5 ] , 
    'min_weight_fraction_leaf': [ 0,0.01,0.05,0.1,0.3,0.5],  # 변경된 설정
    'max_features': ['auto'],           # , 'sqrt', 'log2',None  
    'bootstrap': [True,False],  
}
# param = {
# 'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 10),
# 'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
# 'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
# 'max_depth': trial.suggest_int('max_depth', 5, 100),
# 'min_samples_split': trial.suggest_loguniform('min_samples_split', 0.01, 1.0),
# 'min_samples_leaf': trial.suggest_loguniform('min_samples_leaf', 0.01, 0.5),
# 'min_weight_fraction_leaf': trial.suggest_uniform('min_weight_fraction_leaf', 0.0, 0.5),
# }
# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state= 42,  )

# GridSearchCV 객체 생성
grid_search = GridSearchCV(estimator=rf, param_grid=param_search_space, cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')

# GridSearchCV를 사용한 학습
grid_search.fit(X_train, y_train)

# 최적의 파라미터와 최고 점수 출력
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(best_params)
print(best_score)

""" # RandomizedSearchCV 객체 생성
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_search_space, cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')

# RandomizedSearchCV를 사용한 학습
random_search.fit(X_train, y_train)

# 최적의 파라미터와 최고 점수 출력
best_params = random_search.best_params_
best_score = random_search.best_score_
 """

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

import datetime
dt = datetime.datetime.now()

submit.to_csv(path + 'submit1.csv', index=False)
submit.to_csv(f'C:\_data\dacon\RF/submit_{dt.day}day{dt.hour:2}{dt.minute:2}.csv',index=False)


# {'bootstrap': True, 'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': 47, 'min_impurity_decrease': 0.013264961159866528, 'min_samples_leaf': 0.09170225492671691, 'min_samples_split': 22, 'min_weight_fraction_leaf': 0.14607232426760908, 'n_estimators': 100}
# 0.7648943638865013