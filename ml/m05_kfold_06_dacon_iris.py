from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC


#1
path = "c:/_data/dacon/iris//"

train_csv = pd.read_csv(path + 'train.csv' , index_col = 0)
test_csv = pd.read_csv(path + 'test.csv' , index_col = 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

print(train_csv.shape)          # (120, 5)
print(test_csv.shape)           # (30, 4)


x = train_csv.drop(['species'],axis=1)
y = train_csv['species']


#2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold , cross_val_score , StratifiedKFold

kflod = StratifiedKFold(n_splits=3 , shuffle=True , random_state= 100)

model = RandomForestClassifier()

score = cross_val_score(model,x,y,cv = kflod)
print(score)
print('acc : ',np.mean(score))



# submission_csv['species'] = np.argmax(y_submit, axis=1)
# y_submit도 결과값을 뽑아내야 되는데 그냥 뽑으면 소수점을 나와서 argmax로 위치값의 정수를 뽑아줘야한다.



# submission_csv.to_csv(path + 'submission_0112.csv',index = False)




# arg_test = np.argmax(y_test , axis = 1)
# arg_predict = np.argmax(y_predict , axis = 1)

# def ACC(arg_test,arg_predict):
#     return accuracy_score(arg_test,arg_predict)
# acc = ACC(arg_test,arg_predict)


# print(result)
# print("Acc = ",accuracy_score(y_test,y_predict))

# 0.9444444444444444
# Acc =  0.9444444444444444


# 선의형의 1점


# LinearSVC score  0.9722222222222222
# LinearSVC predict  0.9722222222222222
# Perceptron score  0.8333333333333334
# Perceptron predict  0.8333333333333334
# LogisticRegression score  1.0
# LogisticRegression predict  1.0
# RandomForestClassifier score  1.0
# RandomForestClassifier predict  1.0
# DecisionTreeClassifier score  1.0
# DecisionTreeClassifier predict  1.0
# KNeighborsClassifier score  1.0
# KNeighborsClassifier predict  1.0
# 1.0
# Acc =  1.0


# AdaBoostClassifier 의 Acc는 1.0
# BaggingClassifier 의 Acc는 1.0
# BernoulliNB 의 Acc는 0.3333333333333333
# CalibratedClassifierCV 의 Acc는 0.9722222222222222
# CategoricalNB 의 Acc는 0.9444444444444444
# ClassifierChain 은 바보 멍청이!!!
# ComplementNB 의 Acc는 0.6666666666666666
# DecisionTreeClassifier 의 Acc는 1.0
# DummyClassifier 의 Acc는 0.3333333333333333
# ExtraTreeClassifier 의 Acc는 1.0
# ExtraTreesClassifier 의 Acc는 1.0
# GaussianNB 의 Acc는 1.0
# GaussianProcessClassifier 의 Acc는 1.0
# GradientBoostingClassifier 의 Acc는 1.0
# HistGradientBoostingClassifier 의 Acc는 1.0
# KNeighborsClassifier 의 Acc는 1.0
# LabelPropagation 의 Acc는 1.0
# LabelSpreading 의 Acc는 1.0
# LinearDiscriminantAnalysis 의 Acc는 1.0
# LinearSVC 의 Acc는 0.9722222222222222
# LogisticRegression 의 Acc는 1.0
# LogisticRegressionCV 의 Acc는 1.0
# MLPClassifier 의 Acc는 1.0
# MultiOutputClassifier 은 바보 멍청이!!!
# MultinomialNB 의 Acc는 0.8888888888888888
# NearestCentroid 의 Acc는 0.9444444444444444
# NuSVC 의 Acc는 1.0
# OneVsOneClassifier 은 바보 멍청이!!!
# OneVsRestClassifier 은 바보 멍청이!!!
# OutputCodeClassifier 은 바보 멍청이!!!
# PassiveAggressiveClassifier 의 Acc는 0.7777777777777778
# Perceptron 의 Acc는 0.8333333333333334
# QuadraticDiscriminantAnalysis 의 Acc는 1.0
# RadiusNeighborsClassifier 의 Acc는 1.0
# RandomForestClassifier 의 Acc는 1.0
# RidgeClassifier 의 Acc는 0.8055555555555556
# RidgeClassifierCV 의 Acc는 0.8055555555555556
# SGDClassifier 의 Acc는 0.9444444444444444
# SVC 의 Acc는 1.0
# StackingClassifier 은 바보 멍청이!!!
# VotingClassifier 은 바보 멍청이!!!



# [0.925 0.925 0.95 ]
# acc :  0.9333333333333332

# [0.925 0.975 0.925]
# acc :  0.9416666666666668