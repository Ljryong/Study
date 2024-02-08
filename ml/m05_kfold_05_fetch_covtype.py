from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

#1
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape,y.shape)      # (581012, 54) (581012,)
print(pd.value_counts(y))   # 2    283301 , 1    211840 , 3     35754 , 7     20510 , 6     17367 , 5      9493 , 4      2747   (n,7)




#2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold , cross_val_score , StratifiedKFold

kflod = StratifiedKFold(n_splits=3 , shuffle=True , random_state= 100)

model = RandomForestClassifier()

score = cross_val_score(model,x,y,cv = kflod)
print(score)
print('acc : ',np.mean(score))


# Epoch 35: early stopping
# 5447/5447 [==============================] - 7s 1ms/step - loss: 0.6882 - acc: 0.7010
# 5447/5447 [==============================] - 7s 1ms/step
# (174304,)
# (174304,)
# acc =  0.7009707178263265
# batch = 2500

# 0
# Epoch 22: early stopping
# 5447/5447 [==============================] - 7s 1ms/step - loss: 0.6842 - acc: 0.7016
# 5447/5447 [==============================] - 6s 1ms/step
# (174304,)
# (174304,)
# acc =  0.7015673765375436
# batch = 1000

# 2
# Epoch 22: early stopping
# 5447/5447 [==============================] - 6s 1ms/step - loss: 0.6818 - acc: 0.7041
# 5447/5447 [==============================] - 6s 1ms/step
# (174304,)
# (174304,)
# acc =  0.7041146502662016
# batch = 1000


# MinMaxScaler
# Epoch 93: early stopping
# 5447/5447 [==============================] - 5s 831us/step - loss: 0.6298 - acc: 0.7239
# 5447/5447 [==============================] - 4s 785us/step
# (174304,)
# (174304,)
# acc =  0.7239191297962181
# reslut =  [0.6298187375068665, 0.723919153213501]

# StandardScaler
# Epoch 58: early stopping
# 5447/5447 [==============================] - 4s 806us/step - loss: 0.6309 - acc: 0.7228
# 5447/5447 [==============================] - 4s 806us/step
# (174304,)
# (174304,)
# acc =  0.7227774463007159
# reslut =  [0.6309492588043213, 0.7227774262428284]


# MaxAbsScaler
# Epoch 108: early stopping
# 5447/5447 [==============================] - 5s 842us/step - loss: 0.6302 - acc: 0.7242
# 5447/5447 [==============================] - 4s 804us/step
# (174304,)
# (174304,)
# acc =  0.7242002478428493
# reslut =  [0.630240261554718, 0.7242002487182617]

# RobustScaler
# Epoch 51: early stopping
# 5447/5447 [==============================] - 9s 2ms/step - loss: 0.6287 - acc: 0.7247
# 5447/5447 [==============================] - 8s 1ms/step
# (174304,)
# (174304,)
# acc =  0.7246764273912245
# reslut =  [0.6287251114845276, 0.7246764302253723]


# acc =  0.6820210666421883
# reslut =  0.6820210666421883


# LinearSVC score  0.7131620616853314
# LinearSVC predict  0.7131620616853314
# Perceptron score  0.6213053056728475
# Perceptron predict  0.6213053056728475
# LogisticRegression score  0.7248829630989535
# LogisticRegression predict  0.7248829630989535
# RandomForestClassifier score  0.9530303378006242
# RandomForestClassifier predict  0.9530303378006242
# DecisionTreeClassifier score  0.9359796677069947
# DecisionTreeClassifier predict  0.9359796677069947
# KNeighborsClassifier score  0.9254979805397466
# KNeighborsClassifier predict  0.9254979805397466




# AdaBoostClassifier 의 정답률 0.8703703703703703
# BaggingClassifier 의 정답률 0.9814814814814815
# BernoulliNB 의 정답률 0.3888888888888889
# CalibratedClassifierCV 의 정답률 0.9074074074074074
# CategoricalNB 은 바보 멍충이!!!
# ClassifierChain 은 바보 멍충이!!!
# ComplementNB 의 정답률 0.6296296296296297
# DecisionTreeClassifier 의 정답률 0.9074074074074074
# DummyClassifier 의 정답률 0.3888888888888889
# ExtraTreeClassifier 의 정답률 0.8703703703703703
# ExtraTreesClassifier 의 정답률 1.0
# GaussianNB 의 정답률 0.9629629629629629
# GaussianProcessClassifier 의 정답률 0.42592592592592593
# GradientBoostingClassifier 의 정답률 1.0
# HistGradientBoostingClassifier 의 정답률 1.0
# KNeighborsClassifier 의 정답률 0.7222222222222222
# LabelPropagation 의 정답률 0.4444444444444444
# LabelSpreading 의 정답률 0.4444444444444444
# LinearDiscriminantAnalysis 의 정답률 1.0
# LinearSVC 의 정답률 0.9259259259259259
# LogisticRegression 의 정답률 0.9444444444444444
# LogisticRegressionCV 의 정답률 0.9074074074074074
# MLPClassifier 의 정답률 0.6111111111111112
# MultiOutputClassifier 은 바보 멍충이!!!
# MultinomialNB 의 정답률 0.8148148148148148
# NearestCentroid 의 정답률 0.6851851851851852
# NuSVC 의 정답률 0.8518518518518519
# OneVsOneClassifier 은 바보 멍충이!!!
# OneVsRestClassifier 은 바보 멍충이!!!
# OutputCodeClassifier 은 바보 멍충이!!!
# PassiveAggressiveClassifier 의 정답률 0.7037037037037037
# Perceptron 의 정답률 0.5370370370370371
# QuadraticDiscriminantAnalysis 의 정답률 0.9814814814814815
# RadiusNeighborsClassifier 은 바보 멍충이!!!
# RandomForestClassifier 의 정답률 1.0
# RidgeClassifier 의 정답률 1.0
# RidgeClassifierCV 의 정답률 1.0
# SGDClassifier 의 정답률 0.5925925925925926
# SVC 의 정답률 0.6481481481481481
# StackingClassifier 은 바보 멍충이!!!
# VotingClassifier 은 바보 멍충이!!!


# [0.951588   0.95092709 0.95123148]
# acc :  0.9512488554154145



# [0.95122656 0.95164996 0.95167037]
# acc :  0.9515156316130055