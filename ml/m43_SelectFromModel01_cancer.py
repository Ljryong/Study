import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score , r2_score , mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 1.데이터
x, y =load_breast_cancer(return_X_y=True)

x_train , x_test , y_train , y_test = train_test_split(x,y,random_state=777 , train_size=0.8)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators' : 1000, 
              'learning_rate' : 0.1,
              'max_depth': 3,               # 트리 깊이
              'gamma' : 0,
              'min_child_weight' : 0,       # 드랍 아웃 개념
              'subsample' : 0.4,
              'colsample_bytree' : 0.8,
              'colsample_bylevel' : 0.7,
              'colsample_bynode' : 1,
              'reg_alpha' : 0,              # 알파, 람다 , L1 , L2 규제
              'reg_lamda' : 1,
              'random_state' : 3377,
              'verbose' : 0,
              }

# 2 모델구성
model = XGBClassifier()
model.set_params(early_stopping = 10 , **parameters)

# 3 훈련
model.fit(x_train ,  y_train , eval_set = [(x_train , y_train) , (x_test, y_test)] , verbose = 0 , eval_metric = 'logloss'  )

# 4 평가
results = model.score(x_test,y_test)
print('최종점수', results )

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print('add 스코어',acc)
# add 스코어 0.956140350877193

######################################################################
# for문을 사용해서 피처가 약한놈부터 하나씩 제거
# print(sorted(model.feature_importances_))

thresholds = np.sort(model.feature_importances_)
print(thresholds)
# [0.00126257 0.0056712  0.00611253 0.00739491 0.00808185 0.00832706
#  0.00903037 0.00916403 0.00936591 0.01195777 0.0123292  0.01256799
#  0.01325274 0.01447911 0.01501538 0.01698516 0.01810106 0.02026125
#  0.02465191 0.02465869 0.02647571 0.02698742 0.03868144 0.04184548
#  0.0463003  0.05582664 0.11398669 0.11422888 0.12574384 0.16125299]

print('==================================================================================')
from sklearn.feature_selection import SelectFromModel

for i in thresholds:                                                    # 제일 작은것들을 먼저 없애줌
    # i 보다 크거나 같은 것만 남음 
    selection =  SelectFromModel(model, threshold=i ,prefit=False)        # selectionws은 인스턴스(변수)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(i ,'\t변형된 x_train',select_x_train.shape, i ,'변형된 x_test',select_x_test.shape)
    
    select_model = XGBClassifier()
    select_model.set_params(early_stopping_rounds = 10 , **parameters , eval_metric = 'logloss')
    
    select_model.fit(select_x_train,y_train , eval_set = [(select_x_train , y_train),(select_x_test,y_test)], verbose = 0 ) 
    
    
    select_y_predict = select_model.predict(select_x_test)
    score = accuracy_score(y_test , select_y_predict)
    
    print("Thredsholds=%.3f, n=%d, ACC: %.2f%%" %(i, select_x_train.shape[1], score*100))
    
    
# add 스코어 0.956140350877193
# [0.00126257 0.0056712  0.00611253 0.00739491 0.00808185 0.00832706
#  0.00903037 0.00916403 0.00936591 0.01195777 0.0123292  0.01256799
#  0.01325274 0.01447911 0.01501538 0.01698516 0.01810106 0.02026125
#  0.02465191 0.02465869 0.02647571 0.02698742 0.03868144 0.04184548
#  0.0463003  0.05582664 0.11398669 0.11422888 0.12574384 0.16125299]
# ==================================================================================
# Thredsholds=0.001, n=30, ACC: 95.61%          # Thredsholds == 이건 삭제한 값 feature_importances의 값
# Thredsholds=0.006, n=29, ACC: 92.98%
# Thredsholds=0.006, n=28, ACC: 95.61%
# Thredsholds=0.007, n=27, ACC: 95.61%
# Thredsholds=0.008, n=26, ACC: 94.74%
# Thredsholds=0.008, n=25, ACC: 93.86%
# Thredsholds=0.009, n=24, ACC: 93.86%
# Thredsholds=0.009, n=23, ACC: 92.98%
# Thredsholds=0.009, n=22, ACC: 94.74%
# Thredsholds=0.012, n=21, ACC: 96.49%
# Thredsholds=0.012, n=20, ACC: 93.86%
# Thredsholds=0.013, n=19, ACC: 94.74%
# Thredsholds=0.013, n=18, ACC: 94.74%
# Thredsholds=0.014, n=17, ACC: 95.61%
# Thredsholds=0.015, n=16, ACC: 96.49%
# Thredsholds=0.017, n=15, ACC: 95.61%
# Thredsholds=0.018, n=14, ACC: 94.74%
# Thredsholds=0.020, n=13, ACC: 95.61%
# Thredsholds=0.025, n=12, ACC: 93.86%
# Thredsholds=0.025, n=11, ACC: 95.61%
# Thredsholds=0.026, n=10, ACC: 95.61%
# Thredsholds=0.027, n=9, ACC: 95.61%
# Thredsholds=0.039, n=8, ACC: 93.86%
# Thredsholds=0.042, n=7, ACC: 92.98%
# Thredsholds=0.046, n=6, ACC: 92.98%
# Thredsholds=0.056, n=5, ACC: 92.11%
# Thredsholds=0.114, n=4, ACC: 91.23%
# Thredsholds=0.114, n=3, ACC: 92.11%
# Thredsholds=0.126, n=2, ACC: 90.35%
# Thredsholds=0.161, n=1, ACC: 89.47%