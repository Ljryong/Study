from sklearn.datasets import fetch_california_housing , _california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import time                                 # 시간에 대한 정보를 가져온다
from sklearn.svm import LinearSVR
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

#1
datasets = fetch_california_housing()
# print(datasets.items())
x = datasets.data
y = datasets.target

pf = PolynomialFeatures(degree=2, include_bias=False)
x1 = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size = 0.3, random_state = 59 )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor , VotingRegressor , StackingRegressor
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

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)


# model.score 0.8401449795919554
# True
# model.score 0.6220342782718715
# False
# model.score 0.6127496034406276


# model.score 0.8102075541825241

# model.score 0.8402800944673787

# PolynomialFeatures
# model.score 0.8061424615032632