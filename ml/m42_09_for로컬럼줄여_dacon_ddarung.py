# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np          # 수치 계산이 빠름
import pandas as pd         # 수치 말고 다른 각종 계산들이 좋고 빠름
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.svm import LinearSVR



#1. 데이터

path = "c:/_data/dacon/ddarung//"
# print(path + "aaa_csv") = c:/_data/dacon/ddarung/aaa_csv


train_csv = pd.read_csv(path + "train.csv",index_col = 0) # index_col = 0 , 필요없는 열을 지울 때 사용한다 , index_col = 0 은 0번은 index야 라는 뜻
# \\ 는 2개씩 해야한다 , 하지만 파일 경로일 때는 \ 1개여도 가능                                                                    
# \ \\ / // 다 된다, 섞여도 가능하지만 가독성에 있어서 한개로 하는게 좋다


print(train_csv)     # [1459 rows x 11 columns] = [1459,11] -- index_col = 0 사용하기 전 결과 값

test_csv = pd.read_csv(path + "test.csv", index_col = 0)          # [715 rows x 10 columns] = [715,10] -- index_col = 0 사용하기 전 결과 값
print(test_csv)

submission_csv = pd.read_csv(path + "submission.csv", )   # 서브미션의 index_col을 사용하면 안됨 , 결과 틀에서 벗어날 수 있어서 index_col 을 사용하면 안됨
print(submission_csv)

print(train_csv.shape)      # (1459, 10)
print(test_csv.shape)         # (715, 9)
print(submission_csv.shape)   # (715, 2)            test 랑 submission 2개가 id가 중복된다.

print(train_csv.columns)        
# #Index(['id', 'hour', 'hour_bef_temperature', 'hour_bef_precipitation',
# 'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
# 'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
print(train_csv.info())
#      Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64
# dtypes: float64(9), int64(1)
print(test_csv.info())
#      Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    715 non-null    int64
#  1   hour_bef_temperature    714 non-null    float64
#  2   hour_bef_precipitation  714 non-null    float64
#  3   hour_bef_windspeed      714 non-null    float64
#  4   hour_bef_humidity       714 non-null    float64
#  5   hour_bef_visibility     714 non-null    float64
#  6   hour_bef_ozone          680 non-null    float64
#  7   hour_bef_pm10           678 non-null    float64
#  8   hour_bef_pm2.5          679 non-null    float64
# dtypes: float64(8), int64(1)

print(train_csv.describe())         # describe는 함수이다 , 함수 뒤에는 괄호가 붙는다. 수치 값을 넣어야 사용할 수 있기 때문에 괄호를 붙여야 된다.

######### 결측치 처리 ###########
# 1.제거
'''
print(train_csv.isnull().sum())             # isnull 이랑 isna 똑같다
# print(train_csv.isna().sum())
train_csv = train_csv.dropna()              # 결측치가 1행에 1개라도 있으면 행이 전부 삭제된다
# print(train_csv.info())                   # 결측치 확인 방법
print(train_csv.shape)                      # (1328, 10)      행무시, 열우선
                                            # test data는 결측치를 제거하는 것을 넣으면 안된다. test data는 0이나 mean 값을 넣어줘야 한다.
'''

# 결측치 평균값으로 바꾸는 법
# train_csv = train_csv.fillna(train_csv.mean())  

test_csv = test_csv.fillna(test_csv.mean())                    # 717 non-null     



##################### 결측치를 0으로 바꾸는 법#######################

train_csv = train_csv.fillna(0)

                                          

######### x 와 y 를 분리 ############
x = train_csv.drop(['count'],axis = 1)                # 'count'를 drop 해주세요 axis =1 에서 (count 행(axis = 1)을 drop 해주세요) // 원본을 건드리는 것이 아니라 이 함수만 해당
print(x)
y = train_csv['count']                                # count 만 가져오겠다
print(y)



x_train, x_test, y_train , y_test = train_test_split(x,y,test_size = 0.3, random_state= 846 ) #45

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import RandomizedSearchCV , KFold
kfold = KFold(n_splits= 5 , random_state= 777 , shuffle=True )

#2

from xgboost import XGBRegressor
model = XGBRegressor(tree_method = 'gpu_hist' , random_state = 40 ) 
                           
#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
print(x.shape)
# print('매개변수' , model.best_estimator_ )
# print('매개변수' , model.best_params_ )

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



# n_components =  1 result 0.38212236685918033
# ==================================================
# n_components =  2 result 0.5080723580369471
# ==================================================
# n_components =  3 result 0.56955833898043
# ==================================================
# n_components =  4 result 0.6048110695605808
# ==================================================
# n_components =  5 result 0.6640384714587129
# ==================================================
# n_components =  6 result 0.6949577403612335
# ==================================================
# n_components =  7 result 0.6979739502436869
# ==================================================
# n_components =  8 result 0.6994900687800266
# ==================================================
# n_components =  9 result 0.6983953567913148
# ==================================================


# model.score 0.740383942981283
# (1459, 9)
# 매개변수 XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=0.7, colsample_bynode=0.3, colsample_bytree=0.3,
#              device=None, early_stopping_rounds=None, enable_categorical=False,
#              eval_metric=None, feature_types=None, gamma=10, grow_policy=None,
#              importance_type=None, interaction_constraints=None,
#              learning_rate=0.1, max_bin=None, max_cat_threshold=None,
#              max_cat_to_onehot=None, max_delta_step=None, max_depth=5,
#              max_leaves=None, min_child_weight=5, missing=nan,
#              monotone_constraints=None, multi_strategy=None, n_estimater=400,
#              n_estimators=None, n_jobs=None, num_parallel_tree=None, ...)
# 매개변수 {'subsample': 0.7, 'reg_lambda': 0.01, 'reg_alpha': 0.001, 'n_estimater': 400, 'min_child_weight': 5,
#       'max_depth': 5, 'learning_rate': 0.1, 'gamma': 10, 'colsample_bytree': 0.3, 'colsample_bynode': 0.3, 'colsample_bylevel': 0.7}

# Removing features at indices: [3]
# Accuracy after removing features: 0.8152016050260809
# Total number of removed features: 1

# Removing features at indices: [3 8]
# Accuracy after removing features: 0.8079404878148746
# Total number of removed features: 2

# Removing features at indices: [3 8 5]
# Accuracy after removing features: 0.8037283665220667
# Total number of removed features: 3

# Removing features at indices: [3 8 5 4]
# Accuracy after removing features: 0.7882709732731958
# Total number of removed features: 4

# Removing features at indices: [3 8 5 4 7]
# Accuracy after removing features: 0.7584709938669986
# Total number of removed features: 5

# Removing features at indices: [3 8 5 4 7 6]
# Accuracy after removing features: 0.7428720766160372
# Total number of removed features: 6

# Removing features at indices: [3 8 5 4 7 6 1]
# Accuracy after removing features: 0.6857721868157294
# Total number of removed features: 7

# Removing features at indices: [3 8 5 4 7 6 1 0]
# Accuracy after removing features: 0.0335279973232353
# Total number of removed features: 8