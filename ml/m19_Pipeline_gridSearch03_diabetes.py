# gird(격자)search(찾다)CV(crossvalidation)
# 다 돌려서 가장 좋은 놈을 뻄

import numpy as np
from sklearn.datasets import load_iris , load_digits, load_diabetes
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split , KFold , cross_val_score
from sklearn.experimental import enable_halving_search_cv                   # HalvingGridSearchCV 보다 위에 있어야 돌아감
from sklearn.model_selection import StratifiedKFold ,cross_val_predict , GridSearchCV , RandomizedSearchCV 
from sklearn.model_selection import HalvingGridSearchCV 
from sklearn.metrics import accuracy_score ,r2_score
# cross_val_score 교차검증 스코어
# StratifiedGroupKFold 분류모델의 stratify를 쓰는것
import time
from sklearn.pipeline import make_pipeline , Pipeline
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.ensemble import RandomForestRegressor

#1 데이터
x, y =load_diabetes(return_X_y=True)

kfold = KFold(n_splits=5 , shuffle= True ,random_state=10 )

x_train,x_test , y_train,y_test = train_test_split(x,y,shuffle=True , random_state=123 , test_size=0.2)
print(x_train.shape)                # (353, 10)

parametes = [
    {'svr__C':[1,10,100,1000],'svr__kernel':['linear'],'svr__degree' : [3,4,5]},                                   # 12번 훈련
    {'svr__C':[1,10,100],'svr__kernel':['rbf'],'svr__gamma':[0.001,0.0001]},                                       # 6번 훈련
    {'svr__C':[1,10,100,1000],'svr__kernel':['sigmoid'],'svr__gamma':[0.01,0.001,0.0001],'svr__degree':[3,4]}]          # 24번 훈련
#  총 42개 

# kernel 별로 훈련

#2 모델 구성
# model = SVC(C=1, kernel='linear',degrr=3)
# model = GridSearchCV(SVC(),parametes,cv = kfold,    # 제일 중요한 3개// 얘네가 없으면 안돌아감
#                                                     # 머신러닝종류 , 파라미터 , crossvalidation
#                      verbose=1,
#                      refit=True,                    # 가장 좋은놈만 한번 더 돌려준다.
#                      n_jobs=3                       # 코어 수를 정하는것 , -1일때 전부 사용
                     
#                      )

# model = RandomizedSearchCV(SVC(),parametes,cv = kfold,
#                              verbose=1,
#                              refit=True,
#                              n_jobs= 1 ,
#                              random_state=66,
#                              n_iter= 20             # 반복 횟수 = candidates // default = 10
#                             )

print('===========하빙그리드서치 시작============')
# model = HalvingGridSearchCV(RandomForestRegressor(),parametes,
#                             cv = kfold,
#                              verbose=1,
#                              refit=True,
#                              n_jobs= 1 ,
#                              random_state=66,
#                             #  n_iter= 20,                # 반복 횟수 = candidates // default = 10
#                             factor = 3,                 # Halving 에서 가장 중요함 / default = 3
#                             # min_resources=150           # 
#                             )

# model = make_pipeline(MinMaxScaler(),RandomForestRegressor())

pipe = Pipeline([('st',StandardScaler()) , ('svr' , SVR())])

model = RandomizedSearchCV(pipe , parametes , cv =5 , verbose= 1 , n_jobs= -1 )

start_time = time.time()
model.fit(x_train,y_train)
end_time = time.time()

# print('최적의 매겨번수:' , model.best_estimator_)
# 최적의 매겨번수: SVC(C=1, kernel='linear')
# print('최적의 파라미터:', model.best_params_)
# 파라미터를 뽑을때를 1개만 있어도 뽑을 수 잇다.
# 최적의 파라미터: {'C': 1, 'degree': 3, 'kernel': 'linear'}
# print('best_score' , model.best_score_)
# best_score 0.975
print('model.score' ,  model.score(x_test,y_test))
# model.score 0.9666666666666667

y_predict = model.predict(x_test)
print('r2_score' , r2_score(y_test,y_predict))
# accuracy_score 0.9666666666666667
# y_pred_best = model.best_estimator_.predict(x_test)
            # SVC(C=1, kernel='linear').predict(x_test)
# print('최적의 튠 r2:', r2_score(y_test,y_pred_best))
# 최적의 튠 ACC: 0.9666666666666667
print('걸린시간:' , round(end_time - start_time,2) , '초')


import pandas as pd
print(pd.DataFrame(model.cv_results_).T)


# ===========하빙그리드서치 시작============
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 13
# max_resources_: 353
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 42
# n_resources: 13           # 회귀에서는 CV * 2 + a(알파)
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# ----------
# iter: 1
# n_candidates: 14
# n_resources: 39
# Fitting 5 folds for each of 14 candidates, totalling 70 fits
# ----------
# iter: 2
# n_candidates: 5
# n_resources: 117
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# ----------
# iter: 3
# n_candidates: 2
# n_resources: 351
# Fitting 5 folds for each of 2 candidates, totalling 10 fits


# model.score 0.5261484812827029
# r2_score 0.5261484812827029
# 걸린시간: 0.35 초

# model.score 0.5570011865236196
# r2_score 0.5570011865236196
# 걸린시간: 1.28 초