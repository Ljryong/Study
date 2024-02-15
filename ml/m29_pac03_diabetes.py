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
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# columns = x.columns
# x = pd.DataFrame(x,columns=columns)
# from sklearn.decomposition import PCA
# for i in range(len(x.columns)) :
#     pca = PCA(n_components=i+1)
#     x_train_2 = pca.fit_transform(x_train)
#     x_test_2 = pca.transform(x_test)
#     model = RandomForestRegressor()
#     model.fit(x_train_2,y_train)
#     result = model.score(x_test_2,y_test)
#     print('n_components = ', i+1 ,'result',result)
#     print('='*50)






# ====================================================================================================
# result 0.0
# ====================================================================================================
# result 0.0
# ====================================================================================================
# result 0.007518796992481203
# ====================================================================================================
# result 0.0
# ====================================================================================================
# result 0.015037593984962405
# ====================================================================================================
# result 0.007518796992481203
# ====================================================================================================
# result 0.03007518796992481
# ====================================================================================================
# result 0.022556390977443608
# ====================================================================================================
# result 0.0
# ====================================================================================================
# result 0.007518796992481203