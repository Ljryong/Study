import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , KFold , cross_val_score
from sklearn.model_selection import StratifiedKFold ,cross_val_predict , GridSearchCV , RandomizedSearchCV
from sklearn.metrics import accuracy_score
# cross_val_score 교차검증 스코어
# StratifiedGroupKFold 분류모델의 stratify를 쓰는것
from sklearn.ensemble import RandomForestClassifier
import time

#1 데이터
x, y =load_iris(return_X_y=True)


x_train,x_test , y_train,y_test = train_test_split(x,y,shuffle=True , random_state=123 , test_size=0.2, stratify=y)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True , random_state=123)
# 섞어서 3등분(n_splits=3)으로 나눈다.

from sklearn.preprocessing import MinMaxScaler, StandardScaler , MaxAbsScaler , RobustScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


kfold = StratifiedKFold(n_splits= 3 , shuffle=True , random_state= 1234 )

parameters =[
    {'n_estimators' : [100,200] ,'max_depth':[6,10,12],'min_samples_leaf' : [3,10]},
    {'max_depth': [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3,5,7,10],'min_samples_split' : [2,3,5,10]},
    {'min_samples_split' : [2,3,5,10] },
    {'n_jobs' : [-1,2,4], 'min_samples_split' : [2,3,5,10]}
]
#2 모델

model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold ,
                    verbose=1,
                    refit=True,
                    n_jobs= -1,
                    random_state=66, 
                    n_iter= 10       )

# model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold ,
#                                 verbose=1,
#                                 refit=True,
#                                 n_jobs= -1 )

#3 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4 평가
y_predict = model.predict(x_test)
print('accuracy_score' , accuracy_score(y_test,y_predict))

print(accuracy_score(y_test,y_predict))

print('시간 : ' , round(end - start,2))

# Acc : [1.         0.96666667 0.93333333 1.         0.9       ] 
#  평균 acc : 0.96 

# GridSearchCV
#accuracy_score 0.9666666666666667
# 0.9666666666666667
# 시간 :  2.38

# RandomizedSearchCV
# accuracy_score 0.9666666666666667
# 0.9666666666666667
# 시간 :  1.65