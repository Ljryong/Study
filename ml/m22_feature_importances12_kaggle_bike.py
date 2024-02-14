from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR


#1 데이터

path = 'c:/_data/kaggle/bike//'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')


print(train_csv.shape)      # (10886, 11)

print(test_csv.shape)       # (6493, 8)

print(train_csv.isnull().sum()) 
print(test_csv.isna().sum())

x = train_csv.drop(['casual' , 'registered', 'count'], axis= 1 )        # [6493 rows x 8 columns] // drop을 줄 때 '를 따로 따로 줘야된다.
y = train_csv['count']

print(x)
print(y)            #  10886, 

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state= 6974 ) #7
# x_train_d, x_val , y_train_d, y_val  = train_test_split(x_train, y_train, train_size=0.8, random_state=10)

es = EarlyStopping(monitor = 'val_loss' , mode = 'min', patience = 10 , verbose= 1 ,restore_best_weights=True )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# # scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2 모델구성
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
################### 데이터 프레임 조건 중요 ###################
# print("음수갯수",submission_csv[submission_csv['count']<0].count())    


# plt.figure(figsize = (9,6))
# plt.plot(hist.history['loss'], c = 'red' , label = 'loss' , marker = '.')
# plt.plot(hist.history['val_loss'],c = 'blue' , label = 'val_loss' , marker = '.')
# plt.legend(loc = 'upper right')
# print(hist)

# plt.title('kaggle loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()

# plt.show()

# [6493 rows x 2 columns]
# 로스는 :  [22091.794921875, 22091.794921875, 107.92926025390625]
# 103/103 [==============================] - 0s 503us/step
# R2 =  0.29941292536061126

# # MinMaxScaler
# [6493 rows x 2 columns]
# 로스는 :  [22061.03515625, 22061.03515625, 109.0513916015625]
# 103/103 [==============================] - 0s 717us/step
# R2 =  0.3003884960999149

# # StandardScaler
# [6493 rows x 2 columns]
# 로스는 :  [21999.53515625, 21999.53515625, 109.65789794921875]
# 103/103 [==============================] - 0s 786us/step
# R2 =  0.3023387950309775

# # MaxAbsScaler
# [6493 rows x 2 columns]
# 로스는 :  [21497.927734375, 21497.927734375, 109.61567687988281]
# 103/103 [==============================] - 0s 517us/step
# R2 =  0.31824596802694793

# # RobustScaler
# [6493 rows x 2 columns]
# 로스는 :  [21784.3046875, 21784.3046875, 108.29178619384766]
# 103/103 [==============================] - 0s 582us/step
# R2 =  0.3091640872462358

# r2는 :  0.15482658164940644
# R2 =  0.15482658164940644


# LinearSVR score  0.2126065595439891
# LinearSVR predict  0.2126065595439891
# LinearRegression score  0.24332445147348591
# LinearRegression predict  0.24332445147348591
# RandomForestRegressor score  0.28003914814713615
# RandomForestRegressor predict  0.28003914814713615
# DecisionTreeRegressor score  -0.23003104162624566
# DecisionTreeRegressor predict  -0.23003104162624566
# KNeighborsRegressor score  0.1914773969900625
# KNeighborsRegressor predict  0.1914773969900625

# DecisionTreeRegressor : [0.06371605 0.00771924 0.04262287 0.05703512 0.14257928 0.23421394
#  0.25038134 0.20173217] -0.23212765744203856
# DecisionTreeRegressor result -0.23212765744203856
# RandomForestClassifier : [0.05461152 0.00829953 0.02677142 0.05553354 0.15259008 0.15364025
#  0.27219739 0.27635627] 0.006429883649724434
# RandomForestClassifier result 0.006429883649724434
# GradientBoostingRegressor : [0.07629509 0.00095694 0.0337632  0.01249006 0.18664309 0.3197508
#  0.34543657 0.02466425] 0.3250629865275497
# GradientBoostingRegressor result 0.3250629865275497
# XGBRegressor : [0.12000886 0.05329211 0.09635227 0.07071757 0.10050012 0.34510195
#  0.15045601 0.06357113] 0.3342226715910991
# XGBRegressor result 0.3342226715910991