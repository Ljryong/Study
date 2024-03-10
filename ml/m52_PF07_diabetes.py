from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score
from keras.models import Sequential , load_model
from keras.layers import Dense
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)          # (442, 10) (442,)
print(datasets.feature_names)   #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# x = np.delete(x,(1,7),axis=1)

x = pd.DataFrame(x , columns =datasets.feature_names )
# x = x.drop(['sex', 's4'], axis = 1)

pf = PolynomialFeatures(degree= 2 , include_bias=False )
x1 = pf.fit_transform(x)

x_train , x_test , y_train , y_test = train_test_split(x1,y,test_size = 0.3 , random_state= 151235 , shuffle= True )

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
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor , VotingRegressor, StackingRegressor
from sklearn.linear_model import LogisticRegressionCV , LogisticRegression
from sklearn.svm import LinearSVR
from catboost import CatBoostRegressor

xgb = XGBRegressor()
rf = RandomForestRegressor()
lr = LogisticRegressionCV()
svr = LinearSVR()

model = RandomForestRegressor()


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


# 점수 :  0.3174207067601097
# True
# 점수 :  0.33687448744567405
# False
# 점수 :  0.05808055093258013

# 점수 :  0.364466870941468

# 점수 :  0.2783720353601824

# PolynomialFeatures
# 점수 :  0.36819460710581364