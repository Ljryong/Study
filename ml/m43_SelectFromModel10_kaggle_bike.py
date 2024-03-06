from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR


#1 데이터

path = 'c:/_data/kaggle/bike//'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = train_csv.drop(['casual' , 'registered', 'count'], axis= 1 )        # [6493 rows x 8 columns] // drop을 줄 때 '를 따로 따로 줘야된다.
y = train_csv['count']


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state= 6974 ) #7

es = EarlyStopping(monitor = 'val_loss' , mode = 'min', patience = 10 , verbose= 1 ,restore_best_weights=True )

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

#2 모델구성
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from xgboost import XGBRegressor
model = XGBRegressor(tree_method = 'gpu_hist' , random_state = 40 )

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
# print('매개변수',model.best_estimator_  )
# print('매개변수',model.best_params_  )

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
    
    print("Thredsholds=%.3f, n=%d, R2: %.2f" %(i, select_x_train.shape[1] , score ))


# Thredsholds=0.061, n=8, ACC: 35.92%
# Thredsholds=0.062, n=7, ACC: 35.58%
# Thredsholds=0.074, n=6, ACC: 34.82%
# Thredsholds=0.095, n=5, ACC: 35.01%
# Thredsholds=0.115, n=4, ACC: 33.42%
# Thredsholds=0.126, n=3, ACC: 32.66%
# Thredsholds=0.144, n=2, ACC: 26.44%
# Thredsholds=0.322, n=1, ACC: 17.26%

