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

x_train , x_test, y_train, y_test = train_test_split(x,y,test_size= 0.3 , stratify= y  ,random_state= 1234 , shuffle=True)
#2 모델구성
from sklearn.model_selection import StratifiedKFold , GridSearchCV , RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
kfold = StratifiedKFold(n_splits= 3 , shuffle=True , random_state= 1234 )


#2 모델
model = RandomForestClassifier()
                     

from sklearn.model_selection import StratifiedKFold , cross_val_predict , cross_val_score
kfold = StratifiedKFold(n_splits=5 , shuffle=True , random_state=0)

score = cross_val_score(model , x_train, y_train, cv=kfold  )

print('Acc :',score ,'\n 평균 acc :' , round(score[1],4) )

pred = cross_val_predict(model,x_test,y_test,cv=kfold  )

acc = accuracy_score(y_test,pred)

print(acc)
                     
