import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , KFold , cross_val_score
from sklearn.model_selection import StratifiedKFold ,cross_val_predict
from sklearn.metrics import accuracy_score
# cross_val_score 교차검증 스코어
# StratifiedGroupKFold 분류모델의 stratify를 쓰는것


#1 데이터
x, y =load_iris(return_X_y=True)


x_train,x_test , y_train,y_test = train_test_split(x,y,shuffle=True , random_state=123 , test_size=0.2, stratify=y)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True , random_state=123)
# 섞어서 3등분(n_splits=3)으로 나눈다.

from sklearn.preprocessing import MinMaxScaler, StandardScaler , MaxAbsScaler , RobustScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2 모델구성
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3 훈련
scores = cross_val_score(model, x_train,y_train, cv=kfold )
# cv = cross validation

print('Acc :',scores ,'\n 평균 acc :' , round(np.mean(scores),4) )
# round( ,3) 소수 4번째자리에서 반올림하여 소수 3번째자리까지 나오게함

y_predict = cross_val_predict(model,x_test,y_test,cv=kfold)
print(y_predict)
print(y_test)

acc = accuracy_score(y_test,y_predict)
print('cross_val_predict_Acc',acc)

# Acc : [1.         0.96666667 0.93333333 1.         0.9       ] 
#  평균 acc : 0.96 