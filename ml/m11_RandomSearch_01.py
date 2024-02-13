# gird(격자)search(찾다)CV(crossvalidation)
# 다 돌려서 가장 좋은 놈을 뻄

import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , KFold , cross_val_score
from sklearn.model_selection import StratifiedKFold ,cross_val_predict , GridSearchCV , RandomizedSearchCV
from sklearn.metrics import accuracy_score
# cross_val_score 교차검증 스코어
# StratifiedGroupKFold 분류모델의 stratify를 쓰는것
import time

#1 데이터
x, y =load_iris(return_X_y=True)
kfold = KFold(n_splits=3 , shuffle= True ,random_state=10 )

x_train,x_test , y_train,y_test = train_test_split(x,y,shuffle=True , random_state=123 , test_size=0.2, stratify=y)

parametes = [
    {'C':[1,10,100,1000],'kernel':['linear'],'degree' : [3,4,5]},                                   # 12번 훈련
    {'C':[1,10,100],'kernel':['rbf'],'gamma':[0.001,0.0001]},                                       # 6번 훈련
    {'C':[1,10,100,1000],'kernel':['sigmoid'],'gamma':[0.01,0.001,0.0001],'degree':[3,4]}]          # 24번 훈련
# kernel 별로 훈련

#2 모델 구성
# model = SVC(C=1, kernel='linear',degrr=3)
# model = GridSearchCV(SVC(),parametes,cv = kfold,    # 제일 중요한 3개// 얘네가 없으면 안돌아감
#                                                     # 머신러닝종류 , 파라미터 , crossvalidation
#                      verbose=1,
#                      refit=True,                    # 가장 좋은놈만 한번 더 돌려준다.
#                      n_jobs=3                       # 코어 수를 정하는것 , -1일때 전부 사용
                     
#                      )

model = RandomizedSearchCV(SVC(),parametes,cv = kfold,   
                             verbose=1,
                             refit=True,                   
                             n_jobs= 1 ,
                             random_state=66,        
                             n_iter= 20             # 반복 횟수 = candidates // default = 10
                            )

start_time = time.time()
model.fit(x_train,y_train)
end_time = time.time()

print('최적의 매겨번수:' , model.best_estimator_)
# 최적의 매겨번수: SVC(C=1, kernel='linear')
print('최적의 파라미터:', model.best_params_)
# 파라미터를 뽑을때를 1개만 있어도 뽑을 수 잇다.
# 최적의 파라미터: {'C': 1, 'degree': 3, 'kernel': 'linear'}
print('best_score' , model.best_score_)
# best_score 0.975
print('model.score' ,  model.score(x_test,y_test))
# model.score 0.9666666666666667

y_predict = model.predict(x_test)
print('accuracy_score' , accuracy_score(y_test,y_predict))
# accuracy_score 0.9666666666666667
y_pred_best = model.best_estimator_.predict(x_test)
            # SVC(C=1, kernel='linear').predict(x_test)
print('최적의 튠 ACC:', accuracy_score(y_test,y_pred_best))
# 최적의 튠 ACC: 0.9666666666666667
print('걸린시간:' , round(end_time - start_time,2) , '초')


import pandas as pd
print(pd.DataFrame(model.cv_results_).T)
# 42번 돈걸 확인할 수 있다.



# Fitting 3 folds for each of 10 candidates, totalling 30 fits
# 3개로 나눈걸 10개를 뽑아서 돌리고 총 30번 돈다

