from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

#1

datasets = load_wine()
x= datasets.data
y= datasets.target

print(x.shape,y.shape)      # (178, 13) (178,)
print(pd.value_counts(y))   # 1    71 , 0    59 , 2    48


x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.3, random_state= 0 ,shuffle=True, stratify = y)

es = EarlyStopping(monitor='val_loss', mode = 'min' , verbose= 1 ,patience=20 ,restore_best_weights=True)

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
# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold ,
#                     verbose=1,
#                     refit=True,
#                     n_jobs= -1 )

model = HalvingGridSearchCV(RandomForestClassifier(), parameters, cv = kfold ,
                                verbose=1,
                                refit=True,
                                n_jobs= -1 ,
                                random_state=66,
                                # n_iter=10 ,
                                factor= 3 ,
                                min_resources=20)


#3 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4 평가
y_predict = model.predict(x_test)
print('accuracy_score' , accuracy_score(y_test,y_predict))

print(accuracy_score(y_test,y_predict))

print('시간 : ' , round(end - start,2))

# 결과 0.7037037037037037
# [0 1 0 0 1 2 1 2 1 2 0 1 2 0 2 1 1 1 2 1 0 2 1 1 1 1 1 2 2 1 1 2 1 1 1 1 1
#  1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1]
# accuracy :  0.7037037037037037


# GridSearchCV
# Fitting 3 folds for each of 60 candidates, totalling 180 fits
# accuracy_score 1.0
# 1.0
# 시간 :  2.4

# RandomizedSearchCV
# Fitting 3 folds for each of 10 candidates, totalling 30 fits
# accuracy_score 1.0
# 1.0
# 시간 :  1.69

# ==============================

# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 10
# max_resources_: 398
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 60
# n_resources: 10
# Fitting 3 folds for each of 60 candidates, totalling 180 fits
# ----------
# iter: 1
# n_candidates: 20
# n_resources: 30
# Fitting 3 folds for each of 20 candidates, totalling 60 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 90
# Fitting 3 folds for each of 7 candidates, totalling 21 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 270
# Fitting 3 folds for each of 3 candidates, totalling 9 fits
# accuracy_score 0.9649122807017544
# 0.9649122807017544
# 시간 :  3.33



