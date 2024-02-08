from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

#1

datasets = load_wine()
x= datasets.data
y= datasets.target

print(x.shape,y.shape)      # (178, 13) (178,)
print(pd.value_counts(y))   # 1    71 , 0    59 , 2    48


x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.3, random_state= 0 ,shuffle=True, stratify = y)

es = EarlyStopping(monitor='val_loss', mode = 'min' , verbose= 1 ,patience=20 ,restore_best_weights=True)

#2
x_train , x_test, y_train , y_test = train_test_split(x,y,random_state=123,stratify=y,test_size=0.3, shuffle=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler , MaxAbsScaler , RobustScaler
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#2 모델구성
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

from sklearn.model_selection import StratifiedKFold , cross_val_predict , cross_val_score
kfold = StratifiedKFold(n_splits=5 , shuffle=True , random_state=0)

score = cross_val_score(model , x_train, y_train, cv=kfold  )

print('Acc :',score ,'\n 평균 acc :' , round(score[1],4) )

pred = cross_val_predict(model,x_test,y_test,cv=kfold  )

acc = accuracy_score(y_test,pred)

print(acc)

# 결과 0.7037037037037037
# [0 1 0 0 1 2 1 2 1 2 0 1 2 0 2 1 1 1 2 1 0 2 1 1 1 1 1 2 2 1 1 2 1 1 1 1 1
#  1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1]
# accuracy :  0.7037037037037037


# LinearSVC score  0.8518518518518519
# LinearSVC predict  0.8518518518518519
# Perceptron score  0.5370370370370371
# Perceptron predict  0.5370370370370371
# LogisticRegression score  0.9444444444444444
# LogisticRegression predict  0.9444444444444444
# RandomForestClassifier score  1.0
# RandomForestClassifier predict  1.0
# DecisionTreeClassifier score  0.9629629629629629
# DecisionTreeClassifier predict  0.9629629629629629
# KNeighborsClassifier score  0.7222222222222222
# KNeighborsClassifier predict  0.7222222222222222




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


# [1.         0.96610169 0.98305085]
# acc :  0.9830508474576272


# [0.98333333 0.98305085 0.96610169]
# acc :  0.9774952919020716