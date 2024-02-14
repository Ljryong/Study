from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score
from keras.models import Sequential , load_model
from keras.layers import Dense
import numpy as np
import time
import matplotlib.pyplot as plt

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)          # (442, 10) (442,)
print(datasets.feature_names)   #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

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
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
models = [DecisionTreeRegressor(random_state = 777), RandomForestClassifier(random_state = 777) , 
          GradientBoostingRegressor(random_state = 777),XGBRegressor()]

############## 훈련 반복 for 문 ###################a
for model in models :
    model.fit(x_train,y_train)
    result = model.score(x_test,y_test)
    print(type(model).__name__,':',model.feature_importances_ ,result)
   # y_predict = model.predict(x_test)
    print(type(model).__name__,'result',result)


# DecisionTreeRegressor : [0.05653287 0.01305791 0.22510847 0.05496324 0.05245884 0.04638961
#  0.02551284 0.02079925 0.42968592 0.07549106] -0.17853039646344304
# DecisionTreeRegressor result -0.17853039646344304
# RandomForestClassifier : [0.10828589 0.02619031 0.11663191 0.11935675 0.10912072 0.10962284
#  0.11267109 0.06331304 0.11783489 0.11697255] 0.022556390977443608
# RandomForestClassifier result 0.022556390977443608
# GradientBoostingRegressor : [0.03774045 0.01007067 0.29391017 0.07415588 0.01744333 0.05423725
#  0.02634163 0.02024931 0.39157801 0.07427331] 0.35339168994152614
# GradientBoostingRegressor result 0.35339168994152614
# XGBRegressor : [0.03130212 0.01843802 0.14155771 0.05233907 0.04286493 0.06981438
#  0.04670218 0.08773411 0.44431758 0.06492998] 0.3174207067601097
# XGBRegressor result 0.3174207067601097
