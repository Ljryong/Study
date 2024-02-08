# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np          # 수치 계산이 빠름
import pandas as pd         # 수치 말고 다른 각종 계산들이 좋고 빠름
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.svm import LinearSVR



#1. 데이터

path = "c:/_data/dacon/ddarung//"
# print(path + "aaa_csv") = c:/_data/dacon/ddarung/aaa_csv


train_csv = pd.read_csv(path + "train.csv",index_col = 0) # index_col = 0 , 필요없는 열을 지울 때 사용한다 , index_col = 0 은 0번은 index야 라는 뜻
# \\ 는 2개씩 해야한다 , 하지만 파일 경로일 때는 \ 1개여도 가능                                                                    
# \ \\ / // 다 된다, 섞여도 가능하지만 가독성에 있어서 한개로 하는게 좋다


print(train_csv)     # [1459 rows x 11 columns] = [1459,11] -- index_col = 0 사용하기 전 결과 값

test_csv = pd.read_csv(path + "test.csv", index_col = 0)          # [715 rows x 10 columns] = [715,10] -- index_col = 0 사용하기 전 결과 값
print(test_csv)

submission_csv = pd.read_csv(path + "submission.csv", )   # 서브미션의 index_col을 사용하면 안됨 , 결과 틀에서 벗어날 수 있어서 index_col 을 사용하면 안됨
print(submission_csv)

print(train_csv.shape)      # (1459, 10)
print(test_csv.shape)         # (715, 9)
print(submission_csv.shape)   # (715, 2)            test 랑 submission 2개가 id가 중복된다.

print(train_csv.columns)        
# #Index(['id', 'hour', 'hour_bef_temperature', 'hour_bef_precipitation',
# 'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
# 'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
print(train_csv.info())
#      Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64
# dtypes: float64(9), int64(1)
print(test_csv.info())
#      Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    715 non-null    int64
#  1   hour_bef_temperature    714 non-null    float64
#  2   hour_bef_precipitation  714 non-null    float64
#  3   hour_bef_windspeed      714 non-null    float64
#  4   hour_bef_humidity       714 non-null    float64
#  5   hour_bef_visibility     714 non-null    float64
#  6   hour_bef_ozone          680 non-null    float64
#  7   hour_bef_pm10           678 non-null    float64
#  8   hour_bef_pm2.5          679 non-null    float64
# dtypes: float64(8), int64(1)

print(train_csv.describe())         # describe는 함수이다 , 함수 뒤에는 괄호가 붙는다. 수치 값을 넣어야 사용할 수 있기 때문에 괄호를 붙여야 된다.

######### 결측치 처리 ###########
# 1.제거
'''
print(train_csv.isnull().sum())             # isnull 이랑 isna 똑같다
# print(train_csv.isna().sum())
train_csv = train_csv.dropna()              # 결측치가 1행에 1개라도 있으면 행이 전부 삭제된다
# print(train_csv.info())                   # 결측치 확인 방법
print(train_csv.shape)                      # (1328, 10)      행무시, 열우선
                                            # test data는 결측치를 제거하는 것을 넣으면 안된다. test data는 0이나 mean 값을 넣어줘야 한다.
'''

# 결측치 평균값으로 바꾸는 법
# train_csv = train_csv.fillna(train_csv.mean())  

test_csv = test_csv.fillna(test_csv.mean())                    # 717 non-null     



##################### 결측치를 0으로 바꾸는 법#######################

train_csv = train_csv.fillna(0)

                                          

######### x 와 y 를 분리 ############
x = train_csv.drop(['count'],axis = 1)                # 'count'를 drop 해주세요 axis =1 에서 (count 행(axis = 1)을 drop 해주세요) // 원본을 건드리는 것이 아니라 이 함수만 해당
print(x)
y = train_csv['count']                                # count 만 가져오겠다
print(y)

x_train, x_test, y_train , y_test = train_test_split(x,y,test_size = 0.3, random_state= 846 ) #45

print(x_train.shape, x_test.shape)                    # (1021, 9) (438, 9)---->(929, 9) (399, 9) == 결측치를 제거 했을 때
print(y_train.shape, y_test.shape)                    # (1021,) (438,) ------>(929,) (399,)  == 결측치를 제거 했을 때


#2 모델구성
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold , cross_val_score

model = RandomForestRegressor()

kfold = KFold(n_splits=3 , shuffle=True ,random_state=5)

score = cross_val_score(model,x,y,cv=kfold )

print(score)
print('acc : ' , np.mean(score))



# 로스는 :  2943.575927734375 = 54.2

# 로스는 :  3048.900634765625
# 14/14 [==============================] - 0s 0s/step
# R2 =  0.555676685166897


# 로스는 :  2841.63037109375
# 14/14 [==============================] - 0s 1ms/step
# R2 =  0.5748732131682246

# 로스는 :  2928.133544921875
# 14/14 [==============================] - 0s 1ms/step
# R2 =  0.5972511865433818


# 로스는 :  2953.03564453125
# 14/14 [==============================] - 0s 1ms/step
# R2 =  0.5938260415210248

# 로스는 :  2711.775390625
# 14/14 [==============================] - 0s 0s/step
# R2 =  0.5738032160590182

# 로스는 :  2816.08935546875
# 14/14 [==============================] - 0s 0s/step
# R2 =  0.6120126733503751

# 로스는 :  2663.163818359375
# 14/14 [==============================] - 0s 0s/step
# R2 =  0.6348010541990886

# 로스는 :  2676.828369140625
# 14/14 [==============================] - 0s 1ms/step
# R2 =  0.6329272259885607

# 로스는 :  2652.173583984375
# 14/14 [==============================] - 0s 216us/step
# R2 =  0.6363080952424065



# 로스는 :  0.06708519905577148
# R2 =  0.06708519905577148



# LinearSVR score  0.5262892442851996
# LinearSVR predict  0.5262892442851996
# LinearRegression score  0.6072759286909343
# LinearRegression predict  0.6072759286909343
# RandomForestRegressor score  0.7950294977012764
# RandomForestRegressor predict  0.7950294977012764
# DecisionTreeRegressor score  0.5414605596521274
# DecisionTreeRegressor predict  0.5414605596521274
# KNeighborsRegressor score  0.4068639187657872
# KNeighborsRegressor predict  0.4068639187657872


# ARDRegression  ACC 0.606631484634577
# AdaBoostRegressor  ACC 0.6484007221459966
# BaggingRegressor  ACC 0.7678276816040838
# BayesianRidge  ACC 0.6001985917667967
# CCA 에휴
# DecisionTreeRegressor  ACC 0.5416370898302911
# DummyRegressor  ACC -0.0018446100594635695
# ElasticNet  ACC 0.5876322296212533
# ElasticNetCV  ACC 0.5570929405478964
# ExtraTreeRegressor  ACC 0.6732604467412244
# ExtraTreesRegressor  ACC 0.8153634518561179
# GammaRegressor  ACC 0.46213071529666006
# GaussianProcessRegressor  ACC -1.621719486027318
# GradientBoostingRegressor  ACC 0.7758844481694053
# HistGradientBoostingRegressor  ACC 0.7974671644880429
# HuberRegressor  ACC 0.5674873627501553
# IsotonicRegression 에휴
# KNeighborsRegressor  ACC 0.4068639187657872
# KernelRidge  ACC 0.6052955803976243
# Lars  ACC 0.6072759286909356
# LarsCV  ACC 0.6067406924235009
# Lasso  ACC 0.5938572389297037
# LassoCV  ACC 0.5808122225795715
# LassoLars  ACC 0.31794844275504786
# LassoLarsCV  ACC 0.6067406924235009
# LassoLarsIC  ACC 0.6063677441250521
# LinearRegression  ACC 0.6072759286909343
# LinearSVR  ACC 0.3807753790272196
# MLPRegressor  ACC 0.595837395081829
# MultiOutputRegressor 에휴
# MultiTaskElasticNet 에휴
# MultiTaskElasticNetCV 에휴
# MultiTaskLasso 에휴
# MultiTaskLassoCV 에휴
# NuSVR  ACC 0.0403554615346996
# OrthogonalMatchingPursuit  ACC 0.3788018124015208
# OrthogonalMatchingPursuitCV  ACC 0.5948164021945177
# PLSCanonical 에휴
# PLSRegression  ACC 0.6035002739757391
# PassiveAggressiveRegressor  ACC -0.17484933105574552
# PoissonRegressor  ACC -0.001878252843304562
# QuantileRegressor  ACC 0.41055135947966415
# RANSACRegressor  ACC 0.40213035622324134
# RadiusNeighborsRegressor 에휴
# RandomForestRegressor  ACC 0.7979013775845946
# RegressorChain 에휴
# Ridge  ACC 0.6040962587693928
# RidgeCV  ACC 0.6064510685848754
# SGDRegressor  ACC -5.198182582648889e+25
# SVR  ACC 0.06889340641936148
# StackingRegressor 에휴
# TheilSenRegressor  ACC 0.5897679955039441
# TransformedTargetRegressor  ACC 0.6072759286909343
# TweedieRegressor  ACC 0.5725834898700326
# VotingRegressor 에휴


# [0.78450373 0.79627917 0.75225374]
# acc :  0.7776788799825569



