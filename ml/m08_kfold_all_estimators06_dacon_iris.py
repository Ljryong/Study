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




x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3, random_state= 200 ,shuffle=True , stratify = y )
# OneHot 을 했으면 y가 아니라 OneHot을 넣어줘야한다.
es = EarlyStopping(monitor='val_loss', mode='min' , verbose= 1 , restore_best_weights=True , patience= 1000  )

#2
# model = Sequential()
# model.add(Dense(64,input_dim = 4))
# model.add(Dense(32))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(3, activation= 'softmax'))
# model = LinearSVC(C=100)

#3
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
# model.fit(x_train,y_train, epochs=1000000 , batch_size = 100 , verbose=1, callbacks=[es] , validation_split=0.2 )
# model.fit(x_train,y_train)

#4
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

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