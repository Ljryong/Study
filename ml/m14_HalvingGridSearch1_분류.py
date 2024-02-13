# gird(격자)search(찾다)CV(crossvalidation)
# 다 돌려서 가장 좋은 놈을 뻄

import numpy as np
from sklearn.datasets import load_iris , load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , KFold , cross_val_score
from sklearn.experimental import enable_halving_search_cv                   # HalvingGridSearchCV 보다 위에 있어야 돌아감
from sklearn.model_selection import StratifiedKFold ,cross_val_predict , GridSearchCV , RandomizedSearchCV 
from sklearn.model_selection import HalvingGridSearchCV 
from sklearn.metrics import accuracy_score
# cross_val_score 교차검증 스코어
# StratifiedGroupKFold 분류모델의 stratify를 쓰는것
import time

#1 데이터
x, y =load_iris(return_X_y=True)
x, y =load_digits(return_X_y=True)

kfold = KFold(n_splits=5 , shuffle= True ,random_state=10 )

x_train,x_test , y_train,y_test = train_test_split(x,y,shuffle=True , random_state=123 , test_size=0.2, stratify=y)
print(x_train.shape)                # (1437, 64)

parametes = [
    {'C':[1,10,100,1000],'kernel':['linear'],'degree' : [3,4,5]},                                   # 12번 훈련
    {'C':[1,10,100],'kernel':['rbf'],'gamma':[0.001,0.0001]},                                       # 6번 훈련
    {'C':[1,10,100,1000],'kernel':['sigmoid'],'gamma':[0.01,0.001,0.0001],'degree':[3,4]}]          # 24번 훈련
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
model = HalvingGridSearchCV(SVC(),parametes,
                            cv = kfold,
                             verbose=1,
                             refit=True,
                             n_jobs= 1 ,
                             random_state=66,
                            #  n_iter= 20,                # 반복 횟수 = candidates // default = 10
                            factor = 3,                 # Halving 에서 가장 중요함 / default = 3
                            min_resources=150           # 
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

# Fitting 3 folds for each of 42 candidates, totalling 126 fits
# n_split = 3 , 파라미터 계산량 = 42 , 총 126번 계산

# Fitting 3 folds for each of 14 candidates, totalling 42 fits
# n_split = 3 , 파라미터 걸러질거 거르고 남은 계산량 = 14 , 총 42번 계산

# n_iterations: 2                   2번 돌았어 // 2번 돈 이유는 파라미터가 전체 데이터의 개수(120)를 넘길 수 없어서이다. 18+54은 120이 되지 않지만 다음이 170이여서 2번에서 끝남
#                                   조금 넘어가는 상관 없음 크게 넘어가는것만 되지 않는다. 적게 돌리는것보단 조금 더 크게 돌리는게 통상적으로 더 좋다
# n_required_iterations: 4          4번 도는게 좋다     n_required_iterations , n_possible_iterations 가 같은게 좋다. 다르면 데이터 손실이 있다는 뜻 
# n_possible_iterations: 2          설정 한 값으로 몇번 돌 수 있는지(가능한지) 보여준다
# min_resources_: 18                CV(n_split) * 2 * 라벨의 갯수 = min_resources// 개수가 다를 수 있는데 다른 이유는 안에 담겨있는 알고리즘에 의한 것  
# max_resources_: 120               train 의 최대 수량 (최대 훈련 개수) = x_train의 행의 갯수 
# aggressive_elimination: False     
# factor: 3                         분할한 수

# iter: 0                   돈 횟수 0번째
# n_candidates: 42          파라미터 조합(iter마다 factor만큼 나눠짐)
# n_resources: 18           데이터 수(iter마다 factor만큼 곱해짐)
# Fitting 3 folds for each of 42 candidates, totalling 126 fits


# iter: 1                   돈 횟수 1번째
# n_candidates: 14          42번 돈거의 상위 14개 이렇게 나온 이유는 n_candidates / factor 를 한 것이다. 나누어 떨어지지 않으면 반올림한다.
# n_resources: 54           18이 54가 된 이유는 factor 3 이여서 3분할 된걸 3번 계산에서 3이 곱해짐
# Fitting 3 folds for each of 14 candidates, totalling 42 fits
# 최적의 매겨번수: SVC(C=1000, degree=4, gamma=0.001, kernel='sigmoid')