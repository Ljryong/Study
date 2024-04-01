import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import numpy as np
import random as rn
from sklearn.model_selection import train_test_split

SEED1 = 42
SEED2 = 42
SEED3 = 42

tf.random.set_seed(SEED1)  
np.random.seed(SEED2)
rn.seed(SEED3)

data = pd.read_csv('C:\_data\dacon\RF\\train.csv', index_col=0)
submit = pd.read_csv('C:\_data\dacon\RF\sample_submission.csv')

# person_id 컬럼 제거
X_train = data.drop(['login'],axis=1)
Y_train = data['login']

# GridSearchCV를 위한 하이퍼파라미터 설정
param_search_space = {
    'n_estimators': [10,50,100,500],             
    'criterion' : ['gini', 'entropy'],                     
    'max_depth': [None, 5, 10, 20 ],             
    'min_samples_split': [2  , 5, 10 ],      
    'min_samples_leaf': [ 1,2,4 ],        
    # 'min_weight_fraction_leaf' : [0, 0.1, 0.3 , 0.5 ],
    'max_features' : ['auto','sqrt','log2'],
    # 'max_leaf_nodes' : [None,  3, 5 ],
    # 'min_impurity_decrease' : [0, 1, 3, 5 ] ,
    'bootstrap' : [True, False],                
}


""" param_search_space = {
    'n_estimators': [100,300,1000],             
    'criterion' : ['gini','entropy'],                     
    'max_depth': [None, 5, 10, 15 ,30 , 50 , 68],             
    'min_samples_split': [2, 5,  10, 15 ],      
    'min_samples_leaf': [1, 4, 8, 12  ],        
    # 'min_weight_fraction_leaf' : [0, 0.1, 0.2 , 0.3,0.4,0.5 ],
    'max_features' : ['auto', 'sqrt', 'log2', None],
    # 'max_leaf_nodes' : [None, 2, 3, 5 ],      # default 가 제일 좋음?
    'min_impurity_decrease' : [0, 1, 3, 5 ] ,
    'bootstrap' : [True, False],                
}
 """
x_train , x_test , y_train , y_test = train_test_split(X_train,Y_train,test_size = 0.2 , random_state=42 , stratify=Y_train )


# RandomForestClassifier 객체 생성
rf = RandomForestClassifier( random_state = 42 )

# GridSearchCV 객체 생성
grid_search = GridSearchCV(estimator=rf, 
                           param_grid=param_search_space,
                           cv = 3 , n_jobs=-1, verbose=2, scoring='roc_auc')

# GridSearchCV를 사용한 학습
grid_search.fit(x_train, y_train)

# 최적의 파라미터와 최고 점수 출력
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(best_params, best_score)

""" # model = RandomForestClassifier(**best_params)
model = RandomForestClassifier(**best_params)
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
pred_list = model.predict_proba(x_test)[:,1]
print("score: ",score)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test,pred_list)
print("AUC:  ",auc)
 """

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value

import datetime
dt = datetime.datetime.now()
submit.to_csv(f'C:\_data\dacon\RF/submit_{dt.day}day{dt.hour:2}{dt.minute:2}.csv',index=False)
