from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score
from keras.models import Sequential , load_model
from keras.layers import Dense
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)          # (442, 10) (442,)
print(datasets.feature_names)   #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# x = np.delete(x,(1,7),axis=1)

x = pd.DataFrame(x , columns =datasets.feature_names )
# x = x.drop(['sex', 's4'], axis = 1)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 151235 , shuffle= True )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2 모델구성
from xgboost import XGBRegressor

model = XGBRegressor()



#3 훈련
model.fit(x_train,y_train)

#4 평가, 예측
# GridSearchCV 전용
from sklearn.metrics import r2_score
score = model.score(x_test,y_test)
print('='*100)
# print('매개변수 : ' , model.best_estimator_)
# print('매개변수 : ' , model.best_params_)
print('점수 : ' ,score )

# 초기 특성 중요도
import warnings
warnings.filterwarnings('ignore')
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
              verbose=0,
            #   eval_metric='mlogloss',
              early_stopping_rounds=10)
    
    # 모델 평가
    acc = model.score(x_test_removed, y_test)
    print('Accuracy after removing features:', acc)
    
    # 제거된 피처의 개수를 누적
    removed_features += 1
    print(f"Total number of removed features: {removed_features}\n")


# 매개변수 :  XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=0.7, colsample_bynode=0.3, colsample_bytree=0.3,
#              device=None, early_stopping_rounds=None, enable_categorical=False,
#              eval_metric=None, feature_types=None, gamma=10, grow_policy=None,
#              importance_type=None, interaction_constraints=None,
#              learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#              max_cat_to_onehot=None, max_delta_step=None, max_depth=5,
#              max_leaves=None, min_child_weight=5, missing=nan,
#              monotone_constraints=None, multi_strategy=None, n_estimater=400,
#              n_estimators=None, n_jobs=None, num_parallel_tree=None, ...)
# 매개변수 :  {'subsample': 0.7, 'reg_lambda': 0.01, 'reg_alpha': 0.001, 'n_estimater': 400, 'min_child_weight': 5, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 10, 'colsample_bytree': 0.3, 'colsample_bynode': 0.3, 'colsample_bylevel': 0.7}
# 점수 :  0.4329768294404487

# 점수 :  0.3174207067601097
# [1 0 4 6 3 9 5 7 2 8]
# Removing features at indices: [1]
# Accuracy after removing features: 0.34221145400254427
# Total number of removed features: 1

# Removing features at indices: [1 0]
# Accuracy after removing features: 0.30349184716754074
# Total number of removed features: 2

# Removing features at indices: [1 0 4]
# Accuracy after removing features: 0.3137737731162613
# Total number of removed features: 3

# Removing features at indices: [1 0 4 6]
# Accuracy after removing features: 0.3394573125206123
# Total number of removed features: 4

# Removing features at indices: [1 0 4 6 3]
# Accuracy after removing features: 0.2855832075004172
# Total number of removed features: 5

# Removing features at indices: [1 0 4 6 3 9]
# Accuracy after removing features: 0.27105126482161346
# Total number of removed features: 6

# Removing features at indices: [1 0 4 6 3 9 5]
# Accuracy after removing features: 0.3279405249268892
# Total number of removed features: 7

# Removing features at indices: [1 0 4 6 3 9 5 7]
# Accuracy after removing features: 0.3194758896160462
# Total number of removed features: 8

# Removing features at indices: [1 0 4 6 3 9 5 7 2]
# Accuracy after removing features: 0.1258843842011299
# Total number of removed features: 9