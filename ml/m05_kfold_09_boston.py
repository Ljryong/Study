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

datasets = load_boston()



x = datasets.data
y = datasets.target



#2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold , cross_val_score

kflod = KFold(n_splits=3 , shuffle=True , random_state= 100)

model = RandomForestRegressor()

score = cross_val_score(model,x,y,cv = kflod)
print(score)
print('acc : ',np.mean(score))


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




# ARDRegression  ACC 0.7287962320921637
# AdaBoostRegressor  ACC 0.8286942745251505
# BaggingRegressor  ACC 0.8376408912777744
# BayesianRidge  ACC 0.7430281076761356
# CCA 에휴
# DecisionTreeRegressor  ACC 0.7780604798953229
# DummyRegressor  ACC -0.00472641015694486
# ElasticNet  ACC 0.6854331319538922
# ElasticNetCV  ACC 0.6671821407081449
# ExtraTreeRegressor  ACC 0.7395943924871358
# ExtraTreesRegressor  ACC 0.8802003744739909
# GammaRegressor  ACC 0.7132107424627376
# GaussianProcessRegressor  ACC -5.3580577115611865
# GradientBoostingRegressor  ACC 0.8774715035274355
# HistGradientBoostingRegressor  ACC 0.8777019070460528
# HuberRegressor  ACC 0.6237824424567556
# IsotonicRegression 에휴
# KNeighborsRegressor  ACC 0.5346324982526918
# KernelRidge  ACC 0.722887725306876
# Lars  ACC 0.7504214541234289
# LarsCV  ACC 0.7472989115008494
# Lasso  ACC 0.6647196306339387
# LassoCV  ACC 0.6975159274850932
# LassoLars  ACC -0.00472641015694486
# LassoLarsCV  ACC 0.7479165174842877
# LassoLarsIC  ACC 0.7470747287400692
# LinearRegression  ACC 0.7504214541234283
# LinearSVR  ACC 0.555393135811924
# MLPRegressor  ACC 0.6733508536102781
# MultiOutputRegressor 에휴
# MultiTaskElasticNet 에휴
# MultiTaskElasticNetCV 에휴
# MultiTaskLasso 에휴
# MultiTaskLassoCV 에휴
# NuSVR  ACC 0.2068420984481837
# OrthogonalMatchingPursuit  ACC 0.5616742394225225
# OrthogonalMatchingPursuitCV  ACC 0.7018288449886934
# PLSCanonical 에휴
# PLSRegression  ACC 0.7176252238232136
# PassiveAggressiveRegressor  ACC 0.11749889630723154
# PoissonRegressor  ACC 0.7837996866057935
# QuantileRegressor  ACC 0.35584817289598025
# RANSACRegressor  ACC 0.11652948140254349
# RadiusNeighborsRegressor 에휴
# RandomForestRegressor  ACC 0.8445286351532889
# RegressorChain 에휴
# Ridge  ACC 0.7502128544366222
# RidgeCV  ACC 0.7507313729635541
# SGDRegressor  ACC -1.2861309694827184e+26
# SVR  ACC 0.18583322104125377
# StackingRegressor 에휴
# TheilSenRegressor  ACC 0.7155516595286993
# TransformedTargetRegressor  ACC 0.7504214541234283
# TweedieRegressor  ACC 0.6671317686316525
# VotingRegressor 에휴


# [0.80680951 0.89787831 0.85564014]
# acc :  0.8534426516203562
