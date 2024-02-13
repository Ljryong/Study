from sklearn.datasets import load_boston
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import pandas as pd
# warning 뜨는것을 없애는 방법, 하지만 아직 왜 뜨는지 모르니 보는것을 추천
import warnings
warnings.filterwarnings('ignore') 
from sklearn.svm import LinearSVR
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

datasets = load_boston()

x = datasets.data
y = datasets.target

x_train , x_test, y_train, y_test = train_test_split(x,y,test_size= 0.3  ,random_state= 1234 , shuffle=True)
#2 모델구성
from sklearn.model_selection import KFold , GridSearchCV , RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import time

kfold = KFold(n_splits= 3 , shuffle=True , random_state= 1234 )

parameters =[
    {'n_estimators' : [100,200] ,'max_depth':[6,10,12],'min_samples_leaf' : [3,10]},
    {'max_depth': [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3,5,7,10],'min_samples_split' : [2,3,5,10]},
    {'min_samples_split' : [2,3,5,10] },
    {'n_jobs' : [-1,2,4], 'min_samples_split' : [2,3,5,10]}
]

#2 모델
# model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold ,
#                     verbose=1,
#                     refit=True,
#                     n_jobs= -1 )

model = HalvingGridSearchCV(RandomForestRegressor(), parameters, cv = kfold ,
                                verbose=1,
                                refit=True,
                                n_jobs= -1 ,
                                random_state=66,
                                # n_iter=10,
                                factor=3,
                                # min_resources=5
                                )

model = make_pipeline(MinMaxScaler() , RandomForestRegressor() )

#3 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4 평가
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
print('r2_score' , r2_score(y_test,y_predict))

print(r2_score(y_test,y_predict))

print('시간 : ' , round(end - start,2))


# ----------
# iter: 0
# n_candidates: 60
# n_resources: 13
# Fitting 3 folds for each of 60 candidates, totalling 180 fits
# ----------
# iter: 1
# n_candidates: 20
# n_resources: 39
# Fitting 3 folds for each of 20 candidates, totalling 60 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 117
# Fitting 3 folds for each of 7 candidates, totalling 21 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 351
# Fitting 3 folds for each of 3 candidates, totalling 9 fits
# r2_score 0.896165447123189
# 0.896165447123189
# 시간 :  2.75

