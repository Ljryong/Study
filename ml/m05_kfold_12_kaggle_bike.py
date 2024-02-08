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

x = train_csv.drop(['casual' , 'registered', 'count'], axis= 1 )        # [6493 rows x 8 columns] // drop을 줄 때 '를 따로 따로 줘야된다.
y = train_csv['count']


#2 모델구성
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold , cross_val_score

model = RandomForestRegressor()

kfold = KFold(n_splits=3 , shuffle=True ,random_state=5)

score = cross_val_score(model,x,y,cv=kfold )

print(score)
print('acc : ' , np.mean(score))

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



# ARDRegression  ACC 0.24324601646881594
# AdaBoostRegressor  ACC 0.1809877457416249
# BaggingRegressor  ACC 0.2365036240744074
# BayesianRidge  ACC 0.2435058648277899
# CCA 에휴
# DecisionTreeRegressor  ACC -0.21737254560849473
# DummyRegressor  ACC -0.001816031669857665
# ElasticNet  ACC 0.24295060173747351
# ElasticNetCV  ACC 0.24035694359599435
# ExtraTreeRegressor  ACC -0.17503312796061632
# ExtraTreesRegressor  ACC 0.17831806451264842
# GammaRegressor  ACC 0.17004520079772234
# GaussianProcessRegressor  ACC -0.28457140630433964
# GradientBoostingRegressor  ACC 0.3250733210840605
# HistGradientBoostingRegressor  ACC 0.35216581549394366
# HuberRegressor  ACC 0.2307348294099102
# IsotonicRegression 에휴
# KNeighborsRegressor  ACC 0.1914773969900625
# KernelRidge  ACC 0.22634766629146585
# Lars  ACC 0.24332445147348591
# LarsCV  ACC 0.2435847763100819
# Lasso  ACC 0.24386681995805692
# LassoCV  ACC 0.24393230100158092
# LassoLars  ACC -0.001816031669857665
# LassoLarsCV  ACC 0.2435847763100819
# LassoLarsIC  ACC 0.24357408913801504
# LinearRegression  ACC 0.24332445147348591
# LinearSVR  ACC 0.2200489838087103
# MLPRegressor  ACC 0.2741843022621898
# MultiOutputRegressor 에휴
# MultiTaskElasticNet 에휴
# MultiTaskElasticNetCV 에휴
# MultiTaskLasso 에휴
# MultiTaskLassoCV 에휴
# NuSVR  ACC 0.21340234697335503
# OrthogonalMatchingPursuit  ACC 0.14899933837210888
# OrthogonalMatchingPursuitCV  ACC 0.24171805309684113
# PLSCanonical 에휴
# PLSRegression  ACC 0.24043624017407972
# PassiveAggressiveRegressor  ACC 0.167814158529479
# PoissonRegressor  ACC 0.25653096662416885
# QuantileRegressor  ACC 0.1438424913389078
# RANSACRegressor  ACC 0.018385464792911366
# RadiusNeighborsRegressor  ACC -1.40507231581024e+33
# RandomForestRegressor  ACC 0.2749644652578084
# RegressorChain 에휴
# Ridge  ACC 0.2433250567745623
# RidgeCV  ACC 0.24333057167554595
# SGDRegressor  ACC -84793016934531.6
# SVR  ACC 0.1976052242962577
# StackingRegressor 에휴
# TheilSenRegressor  ACC 0.23662135389318606
# TransformedTargetRegressor  ACC 0.24332445147348591
# TweedieRegressor  ACC 0.24114880239044767
# VotingRegressor 에휴




# [0.29435802 0.29028202 0.29075486]
# acc :  0.2917983006030798



