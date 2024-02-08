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

# one_hot = pd.get_dummies(y)

# from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(-1,1)
# ohe = OneHotEncoder()
# ohe.fit(y)
# one_hot = ohe.transform(y).toarray()

# print(one_hot)



x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state= 2 ,shuffle=True, stratify=y ) # 0

es= EarlyStopping(monitor='val_loss' , mode = 'min', verbose= 1 ,patience=10, restore_best_weights=True )


# print(datasets.DESCR)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2
allAlgorisms = all_estimators(type_filter='classifier')

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



# ===================== AdaBoostClassifier =====================
# Acc : [0.83333333 0.92682927 0.92682927]
#  평균 acc : 0.8957
# cross_val_predict_Acc 0.7407407407407407
# ===================== BaggingClassifier =====================
# Acc : [0.83333333 0.97560976 0.97560976]
#  평균 acc : 0.9282
# cross_val_predict_Acc 0.8518518518518519
# ===================== BernoulliNB =====================
# Acc : [0.4047619  0.3902439  0.41463415]
#  평균 acc : 0.4032
# cross_val_predict_Acc 0.3888888888888889
# ===================== CalibratedClassifierCV =====================
# Acc : [0.92857143 0.90243902 0.87804878]
#  평균 acc : 0.903
# cross_val_predict_Acc 0.8148148148148148
# ===================== CategoricalNB =====================
# Acc : [nan nan nan]
#  평균 acc : nan
# CategoricalNB 은 바보 멍충이!!!
# ClassifierChain 은 바보 멍충이!!!
# ===================== ComplementNB =====================
# Acc : [0.64285714 0.63414634 0.68292683]
#  평균 acc : 0.6533
# cross_val_predict_Acc 0.7037037037037037
# ===================== DecisionTreeClassifier =====================
# Acc : [0.83333333 0.95121951 0.87804878]
#  평균 acc : 0.8875
# cross_val_predict_Acc 0.8148148148148148
# ===================== DummyClassifier =====================
# Acc : [0.4047619  0.3902439  0.41463415]
#  평균 acc : 0.4032
# cross_val_predict_Acc 0.3888888888888889
# ===================== ExtraTreeClassifier =====================
# Acc : [0.85714286 0.80487805 0.7804878 ]
#  평균 acc : 0.8142
# cross_val_predict_Acc 0.8148148148148148
# ===================== ExtraTreesClassifier =====================
# Acc : [0.92857143 0.97560976 0.97560976]
#  평균 acc : 0.9599
# cross_val_predict_Acc 1.0
# ===================== GaussianNB =====================
# Acc : [0.95238095 0.95121951 0.97560976]
#  평균 acc : 0.9597
# cross_val_predict_Acc 0.9814814814814815
# ===================== GaussianProcessClassifier =====================
# Acc : [0.52380952 0.46341463 0.43902439]
#  평균 acc : 0.4754
# cross_val_predict_Acc 0.42592592592592593
# ===================== GradientBoostingClassifier =====================
# Acc : [0.85714286 0.92682927 0.87804878]
#  평균 acc : 0.8873
# cross_val_predict_Acc 0.9444444444444444
# ===================== HistGradientBoostingClassifier =====================
# Acc : [0.9047619  0.95121951 0.95121951]
#  평균 acc : 0.9357
# cross_val_predict_Acc 0.3888888888888889
# ===================== KNeighborsClassifier =====================
# Acc : [0.73809524 0.75609756 0.75609756]
#  평균 acc : 0.7501
# cross_val_predict_Acc 0.6851851851851852
# ===================== LabelPropagation =====================
# Acc : [0.42857143 0.48780488 0.36585366]
#  평균 acc : 0.4274
# cross_val_predict_Acc 0.4074074074074074
# ===================== LabelSpreading =====================
# Acc : [0.42857143 0.48780488 0.36585366] 
#  평균 acc : 0.4274
# cross_val_predict_Acc 0.4074074074074074
# ===================== LinearDiscriminantAnalysis =====================
# Acc : [0.95238095 0.97560976 0.87804878]
#  평균 acc : 0.9353
# cross_val_predict_Acc 0.9629629629629629
# ===================== LinearSVC =====================
# Acc : [0.61904762 0.7804878  0.63414634]
#  평균 acc : 0.6779
# cross_val_predict_Acc 0.5555555555555556
# ===================== LogisticRegression =====================
# Acc : [0.85714286 0.92682927 0.92682927]
#  평균 acc : 0.9036
# cross_val_predict_Acc 0.9259259259259259
# ===================== LogisticRegressionCV =====================
# Acc : [0.9047619  0.92682927 0.92682927]
#  평균 acc : 0.9195
# cross_val_predict_Acc 0.9259259259259259
# ===================== MLPClassifier =====================
# Acc : [0.         0.34146341 0.92682927]
#  평균 acc : 0.4228
# cross_val_predict_Acc 0.3888888888888889
# MultiOutputClassifier 은 바보 멍충이!!!
# ===================== MultinomialNB =====================
# Acc : [0.80952381 0.82926829 0.92682927]
#  평균 acc : 0.8552
# cross_val_predict_Acc 0.8333333333333334
# ===================== NearestCentroid =====================
# Acc : [0.73809524 0.68292683 0.80487805]
#  평균 acc : 0.742
# cross_val_predict_Acc 0.6851851851851852
# ===================== NuSVC =====================
# Acc : [0.88095238 0.87804878 0.80487805]
#  평균 acc : 0.8546
# cross_val_predict_Acc 0.7777777777777778
# OneVsOneClassifier 은 바보 멍충이!!!
# OneVsRestClassifier 은 바보 멍충이!!!
# OutputCodeClassifier 은 바보 멍충이!!!
# ===================== PassiveAggressiveClassifier =====================
# Acc : [0.66666667 0.6097561  0.41463415]
#  평균 acc : 0.5637
# cross_val_predict_Acc 0.42592592592592593
# ===================== Perceptron =====================
# Acc : [0.69047619 0.63414634 0.65853659]
#  평균 acc : 0.6611
# cross_val_predict_Acc 0.5
# ===================== QuadraticDiscriminantAnalysis =====================
# Acc : [0.97619048 0.95121951 0.95121951]
#  평균 acc : 0.9595
# cross_val_predict_Acc 0.3888888888888889
# ===================== RadiusNeighborsClassifier =====================
# Acc : [nan nan nan]
#  평균 acc : nan
# RadiusNeighborsClassifier 은 바보 멍충이!!!
# ===================== RandomForestClassifier =====================
# Acc : [0.95238095 0.97560976 0.97560976]
#  평균 acc : 0.9679
# cross_val_predict_Acc 0.9814814814814815
# ===================== RidgeClassifier =====================
# Acc : [0.95238095 1.         0.90243902]
#  평균 acc : 0.9516
# cross_val_predict_Acc 1.0
# ===================== RidgeClassifierCV =====================
# Acc : [0.92857143 0.97560976 0.92682927]
#  평균 acc : 0.9437
# cross_val_predict_Acc 0.9814814814814815
# ===================== SGDClassifier =====================
# Acc : [0.5        0.6097561  0.73170732]
#  평균 acc : 0.6138
# cross_val_predict_Acc 0.48148148148148145
# ===================== SVC =====================
# Acc : [0.73809524 0.65853659 0.68292683]
#  평균 acc : 0.6932
# cross_val_predict_Acc 0.6666666666666666
# StackingClassifier 은 바보 멍충이!!!
# VotingClassifier 은 바보 멍충이!!!