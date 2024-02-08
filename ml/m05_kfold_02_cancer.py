import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score , mean_squared_error , mean_squared_log_error , accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split , KFold, cross_val_score
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC , SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

#1 데이터                     ####################################      2진 분류        ###########################################
datasets = load_breast_cancer()
# print(datasets)     
print(datasets.DESCR) # datasets 에 있는 describt 만 뽑아줘
print(datasets.feature_names)       # 컬럼 명들이 나옴

x = datasets.data       # x,y 를 정하는 것은 print로 뽑고 data의 이름과 target 이름을 확인해서 적는 것
y = datasets.target
print(x.shape,y.shape)  # (569, 30) (569,)


#2 모델구성
model = SVC()

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5 , shuffle=True , random_state=0)

score = cross_val_score(model , x, y, cv=kfold  )

print('Acc :',score ,'\n 평균 acc :' , round(score[1],4) )


# hist = model.fit(x_train , y_train,epochs = 1000000 , batch_size = 1 ,  validation_split= 0.2 , callbacks=[es] ,)

#4 평가, 예측
# loss = model.evaluate(x_test,y_test)        # evaluate = predict로 훈련한 x_test 값을 y_test 값이랑 비교하여 평가한다.

# print(np.round(y_predict))                  # round = 반올림 시켜주는 것


# def ACC(aaa, bbb) :                                  # aaa,bbb 가 값이 들어가 있는 것이 아니라 '빈 박스' 같은 느낌이다.
#     return np.sqrt(mean_squared_error(aaa, bbb))     
# acc = ACC(y_test,y_predict)                          # 빈 박스를 여기 ACC() 로 묶어주고 빈 박스의 이름을 정해준 것이다.
# print("ACC : ", acc)





# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c = 'red' , label = 'loss' , marker='.')
# # c = 'red' , label = 'loss' , marker='.' // c = color , label = 이름 , marker = 1 epoch 당 . 을 찍어주세요
# plt.plot(hist.history['val_loss'], c = 'blue' , label = 'val_loss' , marker='.')

# plt.plot(hist.history['accuracy'], c = 'green' , label = 'accuracy' , marker = '.')

# plt.legend(loc='upper right') # 라벨을 오른쪽 위에 달아주세요
# plt.title('boston loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()

# plt.show()

# print("loss = ",loss)
# print('r2 = ', r2)


# Epoch 21: early stopping
# 6/6 [==============================] - 0s 337us/step - loss: 0.1925 - accuracy: 0.9532 - mse: 0.0526 - mae: 0.1171
# 6/6 [==============================] - 0s 555us/step
# loss =  [0.19248037040233612, 0.9532163739204407, 0.052645422518253326, 0.11708547174930573]
# r2 =  0.773750019420979

# Epoch 100: early stopping
# 6/6 [==============================] - 0s 1ms/step - loss: 0.2468 - accuracy: 0.9123 - mse: 0.0724 - mae: 0.0954
# 6/6 [==============================] - 0s 0s/step
# loss =  [0.2468472272157669, 0.9122806787490845, 0.07242728769779205, 0.09540403634309769]
# r2 =  0.6960611137712087

# Epoch 196: early stopping
# 6/6 [==============================] - 0s 1ms/step - loss: 0.1194 - accuracy: 0.9532 - mse: 0.0359 - mae: 0.0688
# 6/6 [==============================] - 0s 3ms/step
# loss =  [0.11937375366687775, 0.9532163739204407, 0.035903919488191605, 0.0688357949256897]
# r2 =  0.8549507488147026


# MinMaxScaler
# ACC :  0.20191379308357277
# RMSLE :  0.136111044401846
# loss =  [0.5407991409301758, 0.9473684430122375, 0.04076918214559555, 0.04619225487112999]
# r2 =  0.8247895961750011

# StandardScaler
# ACC :  0.21629522817435004
# RMSLE :  0.14992442783510057
# loss =  [12.056479454040527, 0.9532163739204407, 0.046783626079559326, 0.046783626079559326]
# r2 =  0.798941798941799

# MaxAbsScaler
# ACC :  0.1823480652822172
# RMSLE :  0.1261759413013147
# loss =  [0.6252135634422302, 0.9649122953414917, 0.03325081616640091, 0.03857691213488579]
# r2 =  0.8571006558893743

# RobustScaler
# ACC :  0.2243401373378479
# RMSLE :  0.15646226215180234
# loss =  [0.8371858596801758, 0.9473684430122375, 0.05032849311828613, 0.05584089457988739]
# r2 =  0.7837072917060004


# LinearSVC score  0.9415204678362573
# LinearSVC predict  0.9415204678362573

# Perceptron score  0.9532163742690059
# Perceptron predict  0.9532163742690059

# LogisticRegression score  0.9649122807017544
# LogisticRegression predict  0.9649122807017544

# RandomForestClassifier score  0.9590643274853801
# RandomForestClassifier predict  0.9590643274853801

# DecisionTreeClassifier score  0.935672514619883
# DecisionTreeClassifier predict  0.935672514619883

# KNeighborsClassifier score  0.9649122807017544
# KNeighborsClassifier predict  0.9649122807017544



# AdaBoostClassifier 의 정답률 0.9649122807017544

# BaggingClassifier 의 정답률 0.9473684210526315
# BernoulliNB 의 정답률 0.8947368421052632
# CalibratedClassifierCV 의 정답률 0.9590643274853801
# CategoricalNB 은 바보 멍충이!!!
# ClassifierChain 은 바보 멍충이!!!
# ComplementNB 은 바보 멍충이!!!
# DecisionTreeClassifier 의 정답률 0.9181286549707602
# DummyClassifier 의 정답률 0.631578947368421
# ExtraTreeClassifier 의 정답률 0.9064327485380117
# ExtraTreesClassifier 의 정답률 0.9707602339181286
# GaussianNB 의 정답률 0.9122807017543859
# GaussianProcessClassifier 의 정답률 0.9473684210526315
# GradientBoostingClassifier 의 정답률 0.9649122807017544

# HistGradientBoostingClassifier 의 정답률 0.9824561403508771

# KNeighborsClassifier 의 정답률 0.9649122807017544
# LabelPropagation 의 정답률 0.9298245614035088
# LabelSpreading 의 정답률 0.9298245614035088
# LinearDiscriminantAnalysis 의 정답률 0.9707602339181286
# LinearSVC 의 정답률 0.9415204678362573
# LogisticRegression 의 정답률 0.9649122807017544
# LogisticRegressionCV 의 정답률 0.9532163742690059
# MLPClassifier 의 정답률 0.9532163742690059
# MultiOutputClassifier 은 바보 멍충이!!!
# MultinomialNB 은 바보 멍충이!!!
# NearestCentroid 의 정답률 0.9298245614035088
# NuSVC 의 정답률 0.9181286549707602
# OneVsOneClassifier 은 바보 멍충이!!!
# OneVsRestClassifier 은 바보 멍충이!!!
# OutputCodeClassifier 은 바보 멍충이!!!
# PassiveAggressiveClassifier 의 정답률 0.9473684210526315
# Perceptron 의 정답률 0.9532163742690059
# QuadraticDiscriminantAnalysis 의 정답률 0.9532163742690059
# RadiusNeighborsClassifier 은 바보 멍충이!!!
# RandomForestClassifier 의 정답률 0.9649122807017544
# RidgeClassifier 의 정답률 0.9590643274853801
# RidgeClassifierCV 의 정답률 0.9590643274853801
# SGDClassifier 의 정답률 0.9415204678362573

# SVC 의 정답률 0.9824561403508771

# StackingClassifier 은 바보 멍충이!!!
# VotingClassifier 은 바보 멍충이!!!

# Acc : [0.92982456 0.94736842 0.86842105 0.90350877 0.92920354]
#  평균 acc : 0.9157


# Acc : [0.88596491 0.92982456 0.92105263 0.89473684 0.97345133]
#  평균 acc : 0.9298