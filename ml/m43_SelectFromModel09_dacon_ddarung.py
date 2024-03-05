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

from sklearn.metrics import accuracy_score

# 초기 특성 중요도
import warnings
from sklearn.feature_selection import SelectFromModel
warnings.filterwarnings('ignore')
thresholds = np.sort(model.feature_importances_)
print(thresholds)

for i in thresholds:                                                    # 제일 작은것들을 먼저 없애줌
    # i 보다 크거나 같은 것만 남음 
    selection =  SelectFromModel(model, threshold=i ,prefit=False)        # selectionws은 인스턴스(변수)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(i ,'\t변형된 x_train',select_x_train.shape, i ,'변형된 x_test',select_x_test.shape)
    
    select_model = XGBRegressor()
    select_model.set_params(early_stopping_rounds = 10 , **parameters ,
                            # eval_metric = 'logloss'
                            )
    
    select_model.fit(select_x_train,y_train  , eval_set = [(select_x_train , y_train  ),(select_x_test,y_test  )], verbose = 0 ) 
    
    
    select_y_predict = select_model.predict(select_x_test)
    score = r2_score(y_test , select_y_predict)
    
    print("Thredsholds=%.3f, n=%d, ACC: %.2f%%" %(i, select_x_train.shape[1], score*100))

# Thredsholds=0.012, n=9, ACC: 77.30%
# Thredsholds=0.016, n=8, ACC: 77.62%
# Thredsholds=0.024, n=7, ACC: 77.19%
# Thredsholds=0.026, n=6, ACC: 76.00%
# Thredsholds=0.032, n=5, ACC: 75.86%
# Thredsholds=0.033, n=4, ACC: 74.62%
# Thredsholds=0.075, n=3, ACC: 74.57%
# Thredsholds=0.291, n=2, ACC: 66.59%
# Thredsholds=0.490, n=1, ACC: 3.28%