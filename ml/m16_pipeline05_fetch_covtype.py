from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#1
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape,y.shape)      # (581012, 54) (581012,)
print(pd.value_counts(y))   # 2    283301 , 1    211840 , 3     35754 , 7     20510 , 6     17367 , 5      9493 , 4      2747   (n,7)


x_train , x_test, y_train, y_test = train_test_split(x,y,test_size= 0.3 , stratify= y  ,random_state= 1234 , shuffle=True)
#2 모델구성
from sklearn.model_selection import StratifiedKFold , GridSearchCV , RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
kfold = StratifiedKFold(n_splits= 3 , shuffle=True , random_state= 1234 )

parameters =[
    {'n_estimators' : [100,200] ,'max_depth':[6,10,12],'min_samples_leaf' : [3,10]},
    {'max_depth': [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3,5,7,10],'min_samples_split' : [2,3,5,10]},
    {'min_samples_split' : [2,3,5,10] },
    {'n_jobs' : [-1,2,4], 'min_samples_split' : [2,3,5,10]}
]

#2 모델
model = HalvingGridSearchCV(RandomForestClassifier(), parameters, cv=kfold ,
                    verbose=1,
                    refit=True,
                    n_jobs= -1 ,
                    factor=5,)
                    # min_resources=)

# model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold ,
#                                 verbose=1,
#                                 refit=True,
#                                 n_jobs= -1 ,
#                                 random_state=66,
#                                 n_iter=10)

model = make_pipeline(StandardScaler(), RandomForestClassifier() )

#3 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4 평가
y_predict = model.predict(x_test)
print('accuracy_score' , accuracy_score(y_test,y_predict))

print(accuracy_score(y_test,y_predict))

print('시간 : ' , round(end - start,2))


# GridSearchCV
# Fitting 3 folds for each of 60 candidates, totalling 180 fits
# accuracy_score 0.9528123278869102
# 0.9528123278869102
# 시간 :  1258.29

# RandomizedSearchCV
# Fitting 3 folds for each of 10 candidates, totalling 30 fits
# accuracy_score 0.9526287405911511
# 0.9526287405911511
# 시간 :  264.38

# ----------
# iter: 0
# n_candidates: 60
# n_resources: 16268
# Fitting 3 folds for each of 60 candidates, totalling 180 fits
# ----------
# iter: 1
# n_candidates: 12
# n_resources: 81340
# Fitting 3 folds for each of 12 candidates, totalling 36 fits
# ----------
# iter: 2
# n_candidates: 3
# n_resources: 406700
# Fitting 3 folds for each of 3 candidates, totalling 9 fits
# accuracy_score 0.953173765375436
# 0.953173765375436
# 시간 :  229.12



# accuracy_score 0.9528639618138425
# 0.9528639618138425