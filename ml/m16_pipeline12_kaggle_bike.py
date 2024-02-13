from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


#1 데이터

path = 'c:/_data/kaggle/bike//'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = train_csv.drop(['casual' , 'registered', 'count'], axis= 1 )        # [6493 rows x 8 columns] // drop을 줄 때 '를 따로 따로 줘야된다.
y = train_csv['count']



x_train , x_test, y_train, y_test = train_test_split(x,y,test_size= 0.3 ,random_state= 1234 , shuffle=True)

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
                                )

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
model = make_pipeline(StandardScaler() , RandomForestRegressor())

#3 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4 평가
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
print('accuracy_score' , r2_score(y_test,y_predict))

print(r2_score(y_test,y_predict))

print('시간 : ' , round(end - start,2))


# ----------
# iter: 1
# n_candidates: 20
# n_resources: 846
# Fitting 3 folds for each of 20 candidates, totalling 60 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 2538
# Fitting 3 folds for each of 7 candidates, totalling 21 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 7614
# Fitting 3 folds for each of 3 candidates, totalling 9 fits
# accuracy_score 0.3567710132856893
# 0.3567710132856893
# 시간 :  4.78


