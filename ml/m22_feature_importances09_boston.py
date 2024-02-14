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
print(x)
print(x.shape)      #(506, 13)
print(y)
print(y.shape)      #(506,)


############################
# df = pd.DataFrame(x)
# Nan_num = df.isna().sum()
# print(Nan_num)
############################



print(datasets.feature_names)
# 'CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

print(datasets.DESCR)               
#[실습]
# train , test의 비율을 0.7 이상 0.9 이하
# R2 0.62 이상
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 51 )


#2
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

# #5/5 [==============================] - 0s 0s/step - loss: 23.7223
# 5/5 [==============================] - 0s 4ms/step
# R2 :  0.7506910719128115
# 205.5621416568756
# random = 51 , 20,30,50,30,14,7,1


# 0.46253832901414615
# R2 :  0.46253832901414615

# LinearSVR score  0.6581927102931705
# LinearSVR predict  0.6581927102931705
# LinearRegression score  0.7504214541234283
# LinearRegression predict  0.7504214541234283
# RandomForestRegressor score  0.8386826377473715
# RandomForestRegressor predict  0.8386826377473715
# DecisionTreeRegressor score  0.7902653074275752
# DecisionTreeRegressor predict  0.7902653074275752
# KNeighborsRegressor score  0.5346324982526918
# KNeighborsRegressor predict  0.5346324982526918

# DecisionTreeClassifier : [5.94366869e-02 3.40321973e-02 1.47386369e-02 6.37185955e-03
#  3.64736676e-02 3.08707154e-02 2.60374068e-02 8.26687988e-03
#  5.66564943e-03 4.18092842e-01 3.59259337e-01 4.28039970e-04
#  3.26081786e-04] 0.8308548641688128
# DecisionTreeClassifier result 0.8308548641688128
# RandomForestClassifier : [0.09975371 0.02744462 0.0461448  0.01779239 0.08350856 0.09045858
#  0.07148163 0.02676213 0.01678357 0.25906977 0.25925865 0.0005233
#  0.00101828] 0.8030655478939935
# RandomForestClassifier result 0.8030655478939935
# GradientBoostingClassifier : [2.25046046e-02 1.11133921e-01 2.07658694e-05 7.45680490e-04
#  1.55974102e-02 6.82978537e-03 1.38308256e-03 7.12597589e-03
#  1.05160742e-03 3.85698861e-01 4.47876892e-01 3.14140259e-05
#  0.00000000e+00] 0.7505607709562183
# GradientBoostingClassifier result 0.7505607709562183
# XGBClassifier : [0.0450008  0.40969452 0.01239122 0.01614703 0.03369214 0.01808134
#  0.01456504 0.0304932  0.02011074 0.18640582 0.19193324 0.0109551
#  0.01052993] 0.8510010800033231
# XGBClassifier result 0.8510010800033231

