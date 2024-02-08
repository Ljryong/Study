import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , KFold , cross_val_score
from sklearn.model_selection import StratifiedKFold
# cross_val_score 교차검증 스코어
# StratifiedGroupKFold 분류모델의 stratify를 쓰는것
import pandas as pd


#1 데이터
# x, y =load_iris(return_X_y=True)
datasets = load_iris()
df = pd.DataFrame(datasets.data , columns = datasets.feature_names)
# pandas의 dataframe 형식으로 바꿈
print(df)           # [150 rows x 4 columns]



n_splits = 3
# kfold = StratifiedKFold(n_splits=n_splits,shuffle=True , random_state=123)
kfold = KFold(n_splits=n_splits,shuffle=True , random_state=123)
# 섞어서 3등분(n_splits=3)으로 나눈다.

for train_index, val_index in kfold.split(df):
    print('='*100)
    print(train_index,'\n' , val_index)
    print('훈련데이터의 갯수 : ',len(train_index),
          '검증데이터의 갯수 :  ',len(val_index))


'''
#2 모델구성
from sklearn.ensemble import RandomForestClassifier
model = SVC()

#3 훈련
scores = cross_val_score(model, x,y, cv=kfold )
# cv = cross validation

print('Acc :',scores ,'\n 평균 acc :' , round(np.mean(scores),4) )
# round( ,3) 소수 4번째자리에서 반올림하여 소수 3번째자리까지 나오게함

# Acc : [1.         0.96666667 0.93333333 1.         0.9       ] 
#  평균 acc : 0.96
'''