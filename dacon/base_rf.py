import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import numpy as np
import random as rn

SEED1 = 42
SEED2 = 42
SEED3 = 42

tf.random.set_seed(SEED1)  
np.random.seed(SEED2)
rn.seed(SEED3)

data = pd.read_csv('C:\_data\dacon\RF\\train.csv')
submit = pd.read_csv('C:\_data\dacon\RF\sample_submission.csv')

# person_id 컬럼 제거
X_train = data.drop(['person_id', 'login'], axis=1)
y_train = data['login']

# GridSearchCV를 위한 하이퍼파라미터 설정
param_search_space = {
    'n_estimators': [100,300,1000],
    'max_depth': [None, 5, 10, 7 ],
    'min_samples_split': [2, 10,  5, 15 ],
    'min_samples_leaf': [1, 4, 8, 15  ],
    'bootstrap' : [True, False],
    'min_weight_fraction_leaf' : [0],
    # 'max_features' : ['auto', 'sqrt', 'log2', None],
    'min_impurity_decrease' : [0,] ,
    
}

# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=42)

# GridSearchCV 객체 생성
grid_search = GridSearchCV(estimator=rf, 
                           param_grid=param_search_space,
                           cv = 10 , n_jobs=-1, verbose=2, scoring='roc_auc')

# GridSearchCV를 사용한 학습
grid_search.fit(X_train, y_train)

# 최적의 파라미터와 최고 점수 출력
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(best_params, best_score)


# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

submit.to_csv('C:\_data\dacon\RF\\baseline_submit.csv', index=False)