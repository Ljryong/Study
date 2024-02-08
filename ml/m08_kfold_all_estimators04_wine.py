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
allAlgorisms = all_estimators(type_filter='classifier')             # 분류모델
# allAlgorisms = all_estimators(type_filter='regressor')              # 회귀모델

# print('allAlgorisms',allAlgorisms)      # 튜플형태 // 소괄호안에 담겨져있음
# print('모델의 갯수',len(allAlgorisms))          # 41 == 분류모델의 갯수 // 55 == 회귀모델의 갯수

from sklearn.model_selection import cross_val_predict , cross_val_score , StratifiedKFold
kfold = StratifiedKFold(n_splits=3 , random_state=123 , shuffle= True)

for name, algorithm in allAlgorisms : 
    # 에러가 떳을때 밑에 print로 넘어가고 다음이 실행됨
    try:
        #2 모델
        model = algorithm()
        #3 훈련
        scores = cross_val_score(model, x_train,y_train, cv=kfold )
        # cv = cross validation
        print('=====================',name , '=====================')
        print('Acc :',scores ,'\n 평균 acc :' , round(np.mean(scores),4) )
        # round( ,3) 소수 4번째자리에서 반올림하여 소수 3번째자리까지 나오게함
        y_predict = cross_val_predict(model,x_test,y_test,cv=kfold)
        acc = accuracy_score(y_test,y_predict)
        print('cross_val_predict_Acc',acc)
    except:
        print(name,  '은 바보 멍충이!!!')
        continue            # 에러를 무시하고 쭉 진행됨 


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
