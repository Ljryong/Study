from sklearn.datasets import load_boston
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import pandas as pd
# warning 뜨는것을 없애는 방법, 하지만 아직 왜 뜨는지 모르니 보는것을 추천
import warnings
warnings.filterwarnings('ignore') 
from sklearn.svm import LinearSVR

# 현재 사이킷런 버전 1.3.0 보스턴 안됨, 그래서 삭제
# pip uninstall scikit-learn
# pip uninstall scikit-image
# pip uninstall scikit-learn-intelex

# pip install scikit-learn==0.23.2
datasets = load_boston()


print(datasets)
x = datasets.data
y = datasets.target
          
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 51 )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# ###################
# # scaler = MinMaxScaler()
# scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


#2
from sklearn.ensemble import RandomForestRegressor
'''

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
pca = PCA(n_components=11)
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




# n_components =  1 result 0.2606393860264473
# ==================================================
# n_components =  2 result 0.5591105310501272
# ==================================================
# n_components =  3 result 0.6369475614915123
# ==================================================
# n_components =  4 result 0.6798617797423153
# ==================================================
# n_components =  5 result 0.7502642862319684
# ==================================================
# n_components =  6 result 0.7614619051865218
# ==================================================
# n_components =  7 result 0.7585601882885804
# ==================================================
# n_components =  8 result 0.7974488783595539
# ==================================================
# n_components =  9 result 0.8082820409885543
# ==================================================
# n_components =  10 result 0.8384921041552087
# ==================================================
# n_components =  11 result 0.8449458169975381
# ==================================================
# n_components =  12 result 0.8388654286168605
# ==================================================
# n_components =  13 result 0.8380732801352362
# ==================================================


# model.score 0.8413157396802915
# (506, 13)
# [0.46990999 0.5849582  0.67448548 0.74573555 0.80859257 0.86034698
#  0.90021612 0.93151003 0.95226011 0.96954494 0.98278581]