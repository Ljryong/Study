from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import time                                 # 시간에 대한 정보를 가져온다
from sklearn.svm import LinearSVR

#1
datasets = fetch_california_housing()
print(datasets.items())
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape,y.shape)              # x.shape = (20640, 8) y.shpae = (20640,)

print(type(x), type(y))
x


print(datasets.feature_names)       #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'] // feature_names = 특징 이름 
print(datasets.DESCR)               # datasets에 대한 설명

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 59 )

#2
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

allAlgorism = all_estimators(type_filter='regressor')

for name, algorism in allAlgorism:
    try:
        #2 모델구성
        model = algorism()
        #3 훈련
        model.fit(x_train,y_train)
        #4 평가, 예측
        r2 = model.score(x_test,y_test)
        print(name,' ACC' , r2)
    except:
        print(name, '에휴' )

# 145/145 [==============================] - 0s 488us/step - loss: 0.5743
# 194/194 [==============================] - 0s 425us/step - loss: 0.5125
# 194/194 [==============================] - 0s 435us/step
# R2 :  0.6143151723052073
# 걸린시간 :  328.59512186050415
# epochs = 3000 , batch_size = 100 , test_size = 0.3 , 13,30,50,30,14,7,1 , random_size = ?


# mse
# R2 :  0.6070084257218801
# 걸린시간 :  269.12228894233704
# random_state = 59


# mae
# 145/145 [==============================] - 0s 570us/step - loss: 0.5390
# 194/194 [==============================] - 0s 437us/step - loss: 0.5180
# 194/194 [==============================] - 0s 388us/step
# R2 :  0.5776539345065924
# 걸린시간 :  273.5317852497101

# R2 :  0.5718224996809571
# 걸린시간 :  280.1116874217987

# 145/145 [==============================] - 0s 586us/step - loss: 0.5358
# 194/194 [==============================] - 0s 412us/step - loss: 0.5138
# 194/194 [==============================] - 0s 413us/step
# R2 :  0.6007797449416794
# 0.5138373374938965
# 걸린시간 :  256.96805143356323


# 73/73 [==============================] - 0s 440us/step - loss: 0.5391
# 194/194 [==============================] - 0s 450us/step - loss: 0.5176
# 194/194 [==============================] - 0s 397us/step
# R2 :  0.5837401913957216
# 0.5176324248313904
# 걸린시간 :  131.3535294532776
# batch = 200



# R2 :  0.255097739894124
# 0.255097739894124


# LinearSVR score  -0.9201959771921033
# LinearSVR predict  -0.9201959771921033
# LinearRegression score  0.6221882031957897
# LinearRegression predict  0.6221882031957897
# RandomForestRegressor score  0.812863626679852
# RandomForestRegressor predict  0.812863626679852
# DecisionTreeRegressor score  0.6202084838907109
# DecisionTreeRegressor predict  0.6202084838907109
# KNeighborsRegressor score  0.1317616062885476
# KNeighborsRegressor predict  0.1317616062885476




# ARDRegression  ACC 0.6102057195878118
# AdaBoostRegressor  ACC 0.47979687207778876
# BaggingRegressor  ACC 0.7952316975717096
# BayesianRidge  ACC 0.6221169216895835
# CCA 에휴
# DecisionTreeRegressor  ACC 0.621801188068211
# DummyRegressor  ACC -0.00011368459573746215
# ElasticNet  ACC 0.42483862680764883
# ElasticNetCV  ACC 0.5986108396893837
# ExtraTreeRegressor  ACC 0.6276430161596055
# ExtraTreesRegressor  ACC 0.8185927808646641
# GammaRegressor  ACC -0.00011426620275667432
# GaussianProcessRegressor  ACC -2.7467462873247213
# GradientBoostingRegressor  ACC 0.7981291846280436
# HistGradientBoostingRegressor  ACC 0.8433338406722448
# HuberRegressor  ACC 0.5259260879734473
# IsotonicRegression 에휴
# KNeighborsRegressor  ACC 0.1317616062885476
# KernelRidge  ACC 0.5686907496629566
# Lars  ACC 0.6221882031957897
# LarsCV  ACC 0.6204680022478493
# Lasso  ACC 0.2784811202189371
# LassoCV  ACC 0.6027089213127671
# LassoLars  ACC -0.00011368459573746215
# LassoLarsCV  ACC 0.6204680022478493
# LassoLarsIC  ACC 0.6221882031957897
# LinearRegression  ACC 0.6221882031957897
# LinearSVR  ACC -1.443712271128872
# MLPRegressor  ACC -0.0650124247220667
# MultiOutputRegressor 에휴
# MultiTaskElasticNet 에휴
# MultiTaskElasticNetCV 에휴
# MultiTaskLasso 에휴
# MultiTaskLassoCV 에휴
# NuSVR  ACC 0.00889976965638195
# OrthogonalMatchingPursuit  ACC 0.4882114976394648
# OrthogonalMatchingPursuitCV  ACC 0.6121162208733408
# PLSCanonical 에휴
# PLSRegression  ACC 0.5443635080237517
# PassiveAggressiveRegressor  ACC 0.3593600116475085
# PoissonRegressor  ACC 0.46990897022932465
# QuantileRegressor  ACC -0.04381245208913631
# RANSACRegressor  ACC 0.5241920353433406
# RadiusNeighborsRegressor 에휴
# RandomForestRegressor  ACC 0.8151341522720528
# RegressorChain 에휴
# Ridge  ACC 0.6221733927858974
# RidgeCV  ACC 0.6220400784023243
# SGDRegressor  ACC -1.110293700640238e+30
# SVR  ACC -0.01760194336101817
# StackingRegressor 에휴
# TheilSenRegressor  ACC 0.5369505314264564
# TransformedTargetRegressor  ACC 0.6221882031957897
# TweedieRegressor  ACC 0.5027783033815625
# VotingRegressor 에휴