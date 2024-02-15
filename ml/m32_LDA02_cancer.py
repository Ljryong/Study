import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score , mean_squared_error , mean_squared_log_error , accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA 

#1 데이터                     ####################################      2진 분류        ###########################################
datasets = load_breast_cancer()
# print(datasets)     
# print(datasets.DESCR) # datasets 에 있는 describt 만 뽑아줘
# print(datasets.feature_names)       # 컬럼 명들이 나옴

x = datasets.data       # x,y 를 정하는 것은 print로 뽑고 data의 이름과 target 이름을 확인해서 적는 것
y = datasets.target
# print(x.shape,y.shape)  # (569, 30) (569,)

# 0,2,3,9,11,17,18
# x = np.delete(x,(0,2,3,9,11,17,18),axis=1)
# print(x)
# print(x.shape,y.shape)

x = pd.DataFrame(x , columns = datasets.feature_names )

# x = x.drop(['mean radius','mean perimeter', 'mean area' , 'mean fractal dimension' ,'texture error' , 'concave points error' ,'symmetry error' ],axis=1)
# print(x)

es = EarlyStopping(monitor='val_loss' , mode = 'min' , verbose= 1 , patience= 100 ,restore_best_weights=True )

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 0 ,shuffle=True) # 0

# print(np.unique(y)) # [0 1]




# y 안에 있는 array와 해당 array의 개수들을 알 수 있다.
# numpy 방법
# print(np.unique(y, return_counts=True))              # (array([0, 1]), array([212, 357], dtype=int64)) //  0은 212개 1은 357개 


# # pandas 방법
# print(pd.DataFrame(y).value_counts())               # 3개 다 같지만 행렬일 경우 맨 위에것을 쓰고 아닐경우는 통상적으로 짧은 2번째를 쓴다.
# print(pd.value_counts(y))                           
# print(pd.Series(y).value_counts())


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape)            # (398, 23)

#2 모델구성
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


lda = LinearDiscriminantAnalysis(n_components=1)
x_train_2 = lda.fit_transform(x_train,y_train)
x_test_2 = lda.transform(x_test)


model = RandomForestClassifier()
model.fit(x_train_2,y_train)

result = model.score(x_test_2,y_test)
print('='*100)
print('result',result)

evr = lda.explained_variance_ratio_
                # 설명할수있는 변화율
print(evr)
print(sum(evr))

evr_cumsum = np.cumsum(evr)         # 누적시키면서 계속 더해줌
print(evr_cumsum)


# result 0.8362573099415205
# result 0.9064327485380117
# result 0.9122807017543859
# result 0.9181286549707602
# result 0.9181286549707602
# result 0.9298245614035088
# result 0.9239766081871345
# result 0.9064327485380117
# result 0.8947368421052632
# result 0.8947368421052632
# result 0.9122807017543859
# result 0.9064327485380117
# result 0.8888888888888888
# result 0.8830409356725146
# result 0.8947368421052632
# result 0.9064327485380117
# result 0.9064327485380117
# result 0.9005847953216374
# result 0.8947368421052632
# result 0.9005847953216374
# result 0.9122807017543859
# result 0.9005847953216374
# result 0.9181286549707602
# result 0.9005847953216374
# result 0.8947368421052632
# result 0.9005847953216374
# result 0.8830409356725146
# result 0.8947368421052632
# result 0.8888888888888888
# result 0.9005847953216374


# result 0.9590643274853801
# [1.]
# 1.0
# [1.]