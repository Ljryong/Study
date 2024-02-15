
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC


#1 
path = "c:/_data/dacon/wine//"

train_csv = pd.read_csv(path + "train.csv" , index_col= 0)      # index_col : 컬럼을 무시한다. //  index_col= 0 는 0번째 컬럼을 무시한다. 
test_csv = pd.read_csv(path + "test.csv" , index_col= 0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


# print(train_csv)        # [5497 rows x 13 columns]
# print(test_csv)         # [1000 rows x 12 columns]

# ######################## 사이킷런 문자데이터 수치화 ##################
# from sklearn.preprocessing import LabelEncoder      # 문자데이터를 알파벳 순서대로 수치화한다
# lab = LabelEncoder()
# lab.fit(train_csv)
# trainlab_csv = lab.transform(train_csv)
# print(trainlab_csv)


# #####################################################################

####### keras에 있는 데이터 수치화 방법 ##########
train_csv['type'] = train_csv['type'].replace({'white': 0, 'red':1})
test_csv['type'] = test_csv['type'].replace({'white': 0, 'red':1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']
# print(train_csv)
# print(y.shape)          # (5497,1)





x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.3 , random_state= 971 , shuffle=True , stratify= y )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

es = EarlyStopping(monitor='val_loss' , mode = 'min', verbose=1, patience= 100 , restore_best_weights=True )

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# columns = x.columns
# x = pd.DataFrame(x,columns=columns)
# from sklearn.decomposition import PCA
# for i in range(len(x.columns)) :
#     pca = PCA(n_components=i+1)
#     x_train_2 = pca.fit_transform(x_train)
#     x_test_2 = pca.transform(x_test)
#     model = RandomForestClassifier()
#     model.fit(x_train_2,y_train)
#     result = model.score(x_test_2,y_test)
#     print('n_components = ', i+1 ,'result',result)
#     print('='*50)

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

# model = RandomForestRegressor()
model = RandomForestClassifier()

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
print(x.shape)

evr = pca.explained_variance_ratio_

evr_cumsum = np.cumsum(evr)   
print(evr_cumsum)


# ==================================================
# n_components =  1 result 0.4806060606060606
# ==================================================
# n_components =  2 result 0.5127272727272727
# ==================================================
# n_components =  3 result 0.5218181818181818
# ==================================================
# n_components =  4 result 0.5751515151515152
# ==================================================
# n_components =  5 result 0.616969696969697
# ==================================================
# n_components =  6 result 0.6163636363636363
# ==================================================
# n_components =  7 result 0.636969696969697
# ==================================================
# n_components =  8 result 0.636969696969697
# ==================================================
# n_components =  9 result 0.6442424242424243
# ==================================================
# n_components =  10 result 0.6393939393939394
# ==================================================
# n_components =  11 result 0.6575757575757576
# ==================================================
# n_components =  12 result 0.6460606060606061
# ==================================================


# model.score 0.6375757575757576
# (5497, 12)
# [0.31649163 0.52613257 0.6588525  0.73781223 0.79913175 0.84882972
#  0.89473275 0.93698174 0.96677118 0.98792024 0.99779749]