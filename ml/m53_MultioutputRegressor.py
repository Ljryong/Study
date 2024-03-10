import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression , Ridge , Lasso
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

#1. 데이터
x, y = load_linnerud(return_X_y=True)
print(x.shape,y.shape)      # (20, 3) (20, 3)
# 최종값 -> x : [2, 110, 43.] , y :  [138,  33,  68.]

#2 모델
model = RandomForestRegressor()

#3 훈련
model.fit(x,y)

#4 평가 
score = model.score(x,y)
print(score)
y_pred = model.predict(x)
print(model.__class__.__name__,'스코어 : ' ,round(mean_absolute_error(y,y_pred),4 ) )       # 3.6688
print(model.predict([[2, 110, 43]] ))       # [[156.39  34.59  63.18]]

# #2 모델
# model = Ridge()

# #3 훈련
# model.fit(x,y)
# score = model.score(x,y)
# print(score)
# y_pred = model.predict(x)
# print(model.__class__.__name__,'스코어 : ' ,round(mean_absolute_error(y,y_pred),4 ) )       # 7.4569
# print(model.predict([[2, 110, 43]] ))       # [187.32842123  37.0873515   55.40215097]]

# #2 모델
# model = LinearRegression()

# #3 훈련
# model.fit(x,y)
# score = model.score(x,y)
# print(score)
# y_pred = model.predict(x)
# print(model.__class__.__name__,'스코어 : ' ,round(mean_absolute_error(y,y_pred),4 ) )       # 7.4567
# print(model.predict([[2, 110, 43]] ))       # [[187.33745435  37.08997099  55.40216714]]


# #2 모델
# model = XGBRegressor()

# #3 훈련
# model.fit(x,y)
# score = model.score(x,y)
# print(score)
# y_pred = model.predict(x)
# print(model.__class__.__name__,'스코어 : ' ,round(mean_absolute_error(y,y_pred),4 ) )       # 0.0008
# print(model.predict([[2, 110, 43]] ))    # [[138.0005    33.002136  67.99897 ]]

# #2 모델
# model = LGBMRegressor()             # 에러

# #3 훈련
# model.fit(x,y)
# score = model.score(x,y)
# print(score)
# y_pred = model.predict(x)
# print(model.__class__.__name__,'스코어 : ' ,round(mean_absolute_error(y,y_pred),4 ) )         # 에러 
# print(model.predict([[2, 110, 43]] ))         # 에러



#2 모델
# model = CatBoostRegressor()             # 에러

# #3 훈련
# model.fit(x,y)
# score = model.score(x,y)
# print(score)
# y_pred = model.predict(x)
# print(model.__class__.__name__,'스코어 : ' ,round(mean_absolute_error(y,y_pred),4 ) )       # 에러 
# print(model.predict([[2, 110, 43]] ))    # 에러


# #2 모델
# model = MultiOutputRegressor(LGBMRegressor())        

# #3 훈련
# model.fit(x,y)
# score = model.score(x,y)
# print(score)
# y_pred = model.predict(x)
# print(model.__class__.__name__,'스코어 : ' ,round(mean_absolute_error(y,y_pred),4 ) )         # 8.91
# print(model.predict([[2, 110, 43]] ))         # [[178.6  35.4  56.1]]

#2 모델
model = CatBoostRegressor( loss_function= 'MultiRMSE'  )        # 훈련할 때 RMSE 로 훈련을 시킴

#3 훈련
model.fit(x,y)
score = model.score(x,y)
print(score)
y_pred = model.predict(x)
print(model.__class__.__name__,'스코어 : ' ,round(mean_absolute_error(y,y_pred),4 ) )       #   0.2154
print(model.predict([[2, 110, 43]] ))    # [[138.97756017  33.09066774  67.61547996]]

