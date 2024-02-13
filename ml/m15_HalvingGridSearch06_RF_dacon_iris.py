from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


#1
path = "c:/_data/dacon/iris//"

train_csv = pd.read_csv(path + 'train.csv' , index_col = 0)
test_csv = pd.read_csv(path + 'test.csv' , index_col = 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

print(train_csv.shape)          # (120, 5)
print(test_csv.shape)           # (30, 4)


x = train_csv.drop(['species'],axis=1)
y = train_csv['species']


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
# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold ,
#                     verbose=1,
#                     refit=True,
#                     n_jobs= -1 )

model = HalvingGridSearchCV(RandomForestClassifier(), parameters, cv = kfold ,
                                verbose=1,
                                refit=True,
                                n_jobs= -1 ,
                                random_state=66,
                                # n_iter=10,
                                factor= 5 ,
                                min_resources=3)


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
# accuracy_score 0.9444444444444444
# 0.9444444444444444
# 시간 :  2.75

# RandomizedSearchCV
# Fitting 3 folds for each of 10 candidates, totalling 30 fits
# accuracy_score 0.9444444444444444
# 0.9444444444444444
# 시간 :  2.07


# ----------
# iter: 0
# n_candidates: 60
# n_resources: 3
# Fitting 3 folds for each of 60 candidates, totalling 180 fits
# ----------
# iter: 1
# n_candidates: 12
# n_resources: 15
# Fitting 3 folds for each of 12 candidates, totalling 36 fits
# ----------
# iter: 2
# n_candidates: 3
# n_resources: 75
# Fitting 3 folds for each of 3 candidates, totalling 9 fits
# accuracy_score 0.9444444444444444
# 0.9444444444444444
# 시간 :  3.25