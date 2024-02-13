from sklearn.datasets import load_digits
import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense 
import time
from sklearn.model_selection import train_test_split , RandomizedSearchCV , GridSearchCV , StratifiedKFold , cross_val_predict , cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# import numpy as np

datasets = load_digits()
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape,y.shape)      # (1797, 64) (1797,)
print(pd.value_counts(y ,sort=False ))          # sort= False 로 하면 숫자가 순서대로 정렬됨   
# 0    178
# 1    182
# 2    177
# 3    183
# 4    181
# 5    182
# 6    181
# 7    179
# 8    174
# 9    180


x_train, x_test , y_train , y_test = train_test_split(x,y,test_size= 0.3 , random_state= 123 , stratify=y , shuffle=True)


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
                    n_jobs= -1 )

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
y_pred_best = model.best_estimator_.predict(x_test)
print('최적의 튠 ACC:', accuracy_score(y_test,y_pred_best))
print(accuracy_score(y_test,y_predict))

print('시간 : ' , round(end - start,2))

# GridSearchCV
# accuracy_score 0.9796296296296296
# 최적의 튠 ACC: 0.9796296296296296
# 0.9796296296296296
# 시간 :  3.3

# RandomizedSearchCV
# Fitting 3 folds for each of 10 candidates, totalling 30 fits
# accuracy_score 0.9814814814814815
# 최적의 튠 ACC: 0.9814814814814815
# 0.9814814814814815
# 시간 :  1.91


