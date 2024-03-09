from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import time                                 # 시간에 대한 정보를 가져온다
from sklearn.svm import LinearSVR
import pandas as pd

#1
datasets = fetch_california_housing()
print(datasets.items())
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 59 )

from sklearn.preprocessing import StandardScaler, RobustScaler


'''
###################
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import RandomForestRegressor

columns = datasets.feature_names
# columns = x.columns
x = pd.DataFrame(x,columns=columns)
from sklearn.decomposition import PCA
for i in range(len(x.columns)) :
    pca = PCA(n_components=i+1)
    x_train_2 = pca.fit_transform(x_train)
    x_test_2 = pca.transform(x_test)
    model = RandomForestRegressor()
    model.fit(x_train_2,y_train)
    result = model.score(x_test_2,y_test)
    print('n_components = ', i+1 ,'result',result)
    print('='*50)

'''

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=8)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
#2
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor

model = RandomForestRegressor()
# model = RandomForestClassifier()

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
print(x.shape)

evr = pca.explained_variance_ratio_

evr_cumsum = np.cumsum(evr)   
print(evr_cumsum)


# n_components =  1 result -0.28710211497111615
# ==================================================
# n_components =  2 result 0.19122223901649837
# ==================================================
# n_components =  3 result 0.3746994061251102
# ==================================================
# n_components =  4 result 0.5812858524375956
# ==================================================
# n_components =  5 result 0.6081246215384929
# ==================================================
# n_components =  6 result 0.6600792725331971
# ==================================================
# n_components =  7 result 0.7262655783180297
# ==================================================
# n_components =  8 result 0.7407074320320721
# ==================================================


# model.score 0.7399965968397997
# (20640, 8)
# [0.25324198 0.48881928 0.64784913 0.77668622 0.90217681 0.9843977
#  0.99428932 1.        ]