from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC


#1 
path = "c:/_data/dacon/wine//"

train_csv = pd.read_csv(path + "train.csv" , index_col= 0)      # index_col : 컬럼을 무시한다. //  index_col= 0 는 0번째 컬럼을 무시한다. 
test_csv = pd.read_csv(path + "test.csv" , index_col= 0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


# print(train_csv)        # [5497 rows x 13 columns]
# print(test_csv)         # [1000 rows x 12 columns]


####### keras에 있는 데이터 수치화 방법 ##########
train_csv['type'] = train_csv['type'].replace({'white': 0, 'red':1})
test_csv['type'] = test_csv['type'].replace({'white': 0, 'red':1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']


#2 모델
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold , cross_val_score , StratifiedKFold

kflod = StratifiedKFold(n_splits=3 , shuffle=True , random_state= 100)

model = RandomForestClassifier()

score = cross_val_score(model,x,y,cv = kflod)
print(score)
print('acc : ',np.mean(score))

# submission_csv['quality'] = np.argmax(y_submit, axis=1)+3       # +3 밖에 써줘야 된다. argmax 전에 쓰면 위치값이 안뽑혀있는 상태이고, 안에 쓰면 소용이 없다.
# # y_submit도 결과값을 뽑아내야 되는데 그냥 뽑으면 소수점을 나와서 argmax로 위치값의 정수를 뽑아줘야한다.

# submission_csv.to_csv(path + 'submission_0112.csv',index = False)

# arg_test = np.argmax(y_test , axis = 1)
# arg_predict = np.argmax(y_predict , axis = 1)

# def ACC(arg_test,arg_predict):
#     return accuracy_score(arg_test,arg_predict)
# acc = ACC(arg_test,arg_predict)


# print(loss)
# print("Acc = ",accuracy_score(y_test,y_predict))


# Epoch 46: early stopping
# 52/52 [==============================] - 0s 942us/step - loss: 1.1259 - acc: 0.5139
# 52/52 [==============================] - 0s 873us/step
# 32/32 [==============================] - 0s 1ms/step
# [1.12594735622406, 0.513939380645752]
# Acc =  0.5139393939393939


# 1
# Epoch 238: early stopping
# 52/52 [==============================] - 0s 868us/step - loss: 1.0999 - acc: 0.5364
# 52/52 [==============================] - 0s 456us/step
# 32/32 [==============================] - 0s 1ms/step
# [1.0999300479888916, 0.5363636612892151]
# Acc =  0.5363636363636364


# Epoch 168: early stopping
# 52/52 [==============================] - 0s 976us/step - loss: 1.1065 - acc: 0.5291
# 52/52 [==============================] - 0s 907us/step
# 32/32 [==============================] - 0s 1ms/step
# [1.1064668893814087, 0.5290908813476562]
# Acc =  0.5290909090909091


# Epoch 178: early stopping
# 52/52 [==============================] - 0s 670us/step - loss: 1.0823 - acc: 0.5339
# 52/52 [==============================] - 0s 749us/step
# 32/32 [==============================] - 0s 588us/step
# [1.0823452472686768, 0.5339394211769104]
# Acc =  0.5339393939393939



# Epoch 158: early stopping
# 52/52 [==============================] - 0s 975us/step - loss: 1.0833 - acc: 0.5352
# 52/52 [==============================] - 0s 1ms/step
# 32/32 [==============================] - 0s 1ms/step
# [1.0832772254943848, 0.5351515412330627]
# Acc =  0.5351515151515152

# Epoch 123: early stopping
# 52/52 [==============================] - 0s 914us/step - loss: 1.0967 - acc: 0.5309
# 52/52 [==============================] - 0s 938us/step
# 32/32 [==============================] - 0s 1ms/step
# [1.0966628789901733, 0.5309090614318848]
# Acc =  0.5309090909090909



# Epoch 68: early stopping
# 52/52 [==============================] - 0s 1ms/step - loss: 1.1319 - acc: 0.5327
# 52/52 [==============================] - 0s 1ms/step
# (1650, 7)
# 32/32 [==============================] - 0s 1ms/step
# [1.131880283355713, 0.5327273011207581]
# Acc =  0.5327272727272727
# PS C:\Study> [1.131880283355713, 0.5327273011207581]
# >> Acc =  0.5327272727272727

# Epoch 1718: early stopping
# 52/52 [==============================] - 0s 1ms/step - loss: 1.0814 - acc: 0.5448
# 52/52 [==============================] - 0s 1ms/step
# (1650, 7)
# 32/32 [==============================] - 0s 1ms/step
# [1.0814285278320312, 0.5448485016822815]
# Acc =  0.5448484848484848



# Epoch 9114: early stopping
# 52/52 [==============================] - 0s 1ms/step - loss: 1.0885 - acc: 0.5424
# 52/52 [==============================] - 0s 1ms/step
# (1650, 7)
# 32/32 [==============================] - 0s 1ms/step
# [1.0885460376739502, 0.5424242615699768]
# Acc =  0.5424242424242425


# 0.47878787878787876
# Acc =  0.47878787878787876

# LinearSVC score  0.43757575757575756
# LinearSVC predict  0.43757575757575756
# Perceptron score  0.32606060606060605
# Perceptron predict  0.32606060606060605
# LogisticRegression score  0.4727272727272727
# LogisticRegression predict  0.4727272727272727
# RandomForestClassifier score  0.6533333333333333
# RandomForestClassifier predict  0.6533333333333333
# DecisionTreeClassifier score  0.5581818181818182
# DecisionTreeClassifier predict  0.5581818181818182
# KNeighborsClassifier score  0.47333333333333333
# KNeighborsClassifier predict  0.47333333333333333


# AdaBoostClassifier 의 Acc는 0.46
# BaggingClassifier 의 Acc는 0.623030303030303
# BernoulliNB 의 Acc는 0.43333333333333335
# CalibratedClassifierCV 의 Acc는 0.4860606060606061
# CategoricalNB 은 바보 멍청이!!!
# ClassifierChain 은 바보 멍청이!!!
# ComplementNB 의 Acc는 0.3703030303030303
# DecisionTreeClassifier 의 Acc는 0.5515151515151515
# DummyClassifier 의 Acc는 0.4393939393939394
# ExtraTreeClassifier 의 Acc는 0.5745454545454546
# ExtraTreesClassifier 의 Acc는 0.6545454545454545
# GaussianNB 의 Acc는 0.4121212121212121
# GaussianProcessClassifier 의 Acc는 0.5309090909090909
# GradientBoostingClassifier 의 Acc는 0.5775757575757576
# HistGradientBoostingClassifier 의 Acc는 0.6351515151515151
# KNeighborsClassifier 의 Acc는 0.47333333333333333
# LabelPropagation 의 Acc는 0.5284848484848484
# LabelSpreading 의 Acc는 0.5284848484848484
# LinearDiscriminantAnalysis 의 Acc는 0.5436363636363636
# LinearSVC 의 Acc는 0.4484848484848485
# LogisticRegression 의 Acc는 0.4727272727272727
# LogisticRegressionCV 의 Acc는 0.5139393939393939
# MLPClassifier 의 Acc는 0.47878787878787876
# MultiOutputClassifier 은 바보 멍청이!!!
# MultinomialNB 의 Acc는 0.3806060606060606
# NearestCentroid 의 Acc는 0.11575757575757575
# NuSVC 은 바보 멍청이!!!
# OneVsOneClassifier 은 바보 멍청이!!!
# OneVsRestClassifier 은 바보 멍청이!!!
# OutputCodeClassifier 은 바보 멍청이!!!
# PassiveAggressiveClassifier 의 Acc는 0.3606060606060606
# Perceptron 의 Acc는 0.32606060606060605
# QuadraticDiscriminantAnalysis 의 Acc는 0.4696969696969697
# RadiusNeighborsClassifier 은 바보 멍청이!!!
# RandomForestClassifier 의 Acc는 0.6430303030303031
# RidgeClassifier 의 Acc는 0.5339393939393939
# RidgeClassifierCV 의 Acc는 0.5339393939393939
# SGDClassifier 의 Acc는 0.41515151515151516
# SVC 의 Acc는 0.44
# StackingClassifier 은 바보 멍청이!!!
# VotingClassifier 은 바보 멍청이!!!


# [0.64811784 0.66539301 0.66375546]
# acc :  0.6590887704076406

# [0.64375341 0.64956332 0.66539301]
# acc :  0.6529032471961952