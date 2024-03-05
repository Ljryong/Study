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

# 초기 특성 중요도
feature_importances = model.feature_importances_
sort= np.argsort(feature_importances)               # argsort 열의 번호로 반환해줌
print(sort)

removed_features = 0

# 각 반복에서 피처를 추가로 제거하면서 성능 평가
for i in range(len(model.feature_importances_) - 1):
    remove = sort[:i+1]  # 추가로 제거할 피처의 인덱스
    
    print(f"Removing features at indices: {remove}")
    
    # 해당 특성 제거
    x_train_removed = np.delete(x_train, remove, axis=1)
    x_test_removed = np.delete(x_test, remove, axis=1)

    # 모델 재구성 및 훈련
    model.fit(x_train_removed, y_train, eval_set=[(x_train_removed, y_train), (x_test_removed, y_test)],
              verbose=0, eval_metric='logloss', early_stopping_rounds=10)
    
    # 모델 평가
    acc = model.score(x_test_removed, y_test)
    print('Accuracy after removing features:', acc)
    
    # 제거된 피처의 개수를 누적
    removed_features += 1
    print(f"Total number of removed features: {removed_features}\n")
