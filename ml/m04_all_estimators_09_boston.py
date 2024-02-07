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
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

allAlgorism = all_estimators(type_filter='regressor')

for name, algofism in allAlgorism:
    try:
        #2 모델구성
        model = allAlgorism()
        #3 훈련
        model.fit(x_train,y_train)
        #4 평가, 예측
        r2 = model.score(x_test,y_test)
    except:
        print(name, '에휴' )



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