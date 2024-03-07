# SMOTE 증폭을 때릴 때 무작정 때리는게 아니라 1000개 10000개 이상이 넘어가는 건 균등한 비율로 잘라서 SMOTE를 때리는게 좋다
from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score
from sklearn.model_selection import KFold , StratifiedKFold
from sklearn.datasets import fetch_covtype

#1
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
y = y - 1

print(x.shape,y.shape)      # (581012, 54) (581012,)
print(pd.value_counts(y))   # 2    283301 , 1    211840 , 3     35754 , 7     20510 , 6     17367 , 5      9493 , 4      2747   (n,7)

x_train , x_test , y_train , y_test = train_test_split(x,y ,test_size=0.3 , random_state= 2222 ,shuffle=True, stratify=y ) # 0

es= EarlyStopping(monitor='val_loss' , mode = 'min', verbose= 1 ,patience=10, restore_best_weights=True )

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0,
              k_neighbors=5,
              )
x_train , y_train  = smote.fit_resample(x_train,y_train)

print(pd.value_counts(y_train))   # 2    283301 , 1    211840 , 3     35754 , 7     20510 , 6     17367 , 5      9493 , 4      2747   (n,7)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier , XGBRegressor
parameter = {'tree_method':'gpu_hist'}
model = XGBClassifier( **parameter , random_state = 777 
                      , n_jobs = -1 
                      )

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
pre = model.predict(x_test)
print('model.score' , result)
print('acc',accuracy_score(y_test,pre))
print('f1',f1_score(y_test,pre, average = 'macro'))

# 사용전
# model.score 0.8719019643840646
# acc 0.8719019643840646
# f1 0.8548565849952636

# SMOTE 사용
# model.score 0.8396823939783367
# acc 0.8396823939783367
# f1 0.8255721658982734