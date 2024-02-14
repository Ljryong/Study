from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import time                                 # 시간에 대한 정보를 가져온다
from sklearn.svm import LinearSVR

#1
datasets = fetch_california_housing()
print(datasets.items())
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape,y.shape)              # x.shape = (20640, 8) y.shpae = (20640,)

print(type(x), type(y))

print(datasets.feature_names)       #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'] // feature_names = 특징 이름 
print(datasets.DESCR)               # datasets에 대한 설명

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 59 )

#2
# model = Sequential()
# model.add(Dense(13,input_dim = 8))
# model.add(Dense(30))
# model.add(Dense(50))
# model.add(Dense(30))
# model.add(Dense(14))
# model.add(Dense(7))
# model.add(Dense(1))
# model = LinearSVR(C=300)

# #3
# # model.compile(loss='mae',optimizer='adam')
# # start_time = time.time()                                    # 시작 시간을 기록
# # model.fit(x_train,y_train,epochs = 3000 , batch_size = 100)
# # end_time = time.time()                                      # 끝나는 시간을 기록
# model.fit(x_train,y_train )

# #4
# loss = model.score(x_test,y_test)
# y_predict = model.predict(x_test)
# r2 = r2_score(y_test,y_predict)
# print('R2 : ',r2)
# print(loss)
# print('걸린시간 : ', end_time - start_time)

# R2 0.55 ~ 0.6 이상

from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
models = [DecisionTreeRegressor(random_state = 777), RandomForestRegressor(random_state = 777) , 
          GradientBoostingRegressor(random_state = 777),XGBRegressor()]

############## 훈련 반복 for 문 ###################a
for model in models :
    model.fit(x_train,y_train)
    result = model.score(x_test,y_test)
    print(type(model).__name__,':',model.feature_importances_ ,result)
   # y_predict = model.predict(x_test)
    print(type(model).__name__,'result',result)

# 145/145 [==============================] - 0s 488us/step - loss: 0.5743
# 194/194 [==============================] - 0s 425us/step - loss: 0.5125
# 194/194 [==============================] - 0s 435us/step
# R2 :  0.6143151723052073
# 걸린시간 :  328.59512186050415
# epochs = 3000 , batch_size = 100 , test_size = 0.3 , 13,30,50,30,14,7,1 , random_size = ?


# mse
# R2 :  0.6070084257218801
# 걸린시간 :  269.12228894233704
# random_state = 59


# mae
# 145/145 [==============================] - 0s 570us/step - loss: 0.5390
# 194/194 [==============================] - 0s 437us/step - loss: 0.5180
# 194/194 [==============================] - 0s 388us/step
# R2 :  0.5776539345065924
# 걸린시간 :  273.5317852497101

# R2 :  0.5718224996809571
# 걸린시간 :  280.1116874217987

# 145/145 [==============================] - 0s 586us/step - loss: 0.5358
# 194/194 [==============================] - 0s 412us/step - loss: 0.5138
# 194/194 [==============================] - 0s 413us/step
# R2 :  0.6007797449416794
# 0.5138373374938965
# 걸린시간 :  256.96805143356323


# 73/73 [==============================] - 0s 440us/step - loss: 0.5391
# 194/194 [==============================] - 0s 450us/step - loss: 0.5176
# 194/194 [==============================] - 0s 397us/step
# R2 :  0.5837401913957216
# 0.5176324248313904
# 걸린시간 :  131.3535294532776
# batch = 200



# R2 :  0.255097739894124
# 0.255097739894124


# LinearSVR score  -0.9201959771921033
# LinearSVR predict  -0.9201959771921033
# LinearRegression score  0.6221882031957897
# LinearRegression predict  0.6221882031957897
# RandomForestRegressor score  0.812863626679852
# RandomForestRegressor predict  0.812863626679852
# DecisionTreeRegressor score  0.6202084838907109
# DecisionTreeRegressor predict  0.6202084838907109
# KNeighborsRegressor score  0.1317616062885476
# KNeighborsRegressor predict  0.1317616062885476

# DecisionTreeRegressor : [0.51634003 0.05476238 0.05228304 0.02703154 0.03465202 0.13102202
#  0.09470404 0.08920491] 0.6135329507134679
# DecisionTreeRegressor result 0.6135329507134679
# RandomForestRegressor : [0.51597553 0.05476037 0.04841532 0.03030627 0.03399598 0.13836251
#  0.08847662 0.0897074 ] 0.8148791767980628
# RandomForestRegressor result 0.8148791767980628
# GradientBoostingRegressor : [0.59208256 0.03218245 0.02323267 0.00450768 0.00363952 0.12827322
#  0.10128557 0.11479635] 0.7981305158921556
# GradientBoostingRegressor result 0.7981305158921556
# XGBRegressor : [0.4787341  0.06750263 0.04521086 0.02521507 0.02525008 0.15376164
#  0.09850606 0.10581949] 0.8401449812488515
# XGBRegressor result 0.8401449812488515