from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold , StratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

#1
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
y = y - 1

print(x.shape,y.shape)      # (581012, 54) (581012,)
print(pd.value_counts(y))   # 2    283301 , 1    211840 , 3     35754 , 7     20510 , 6     17367 , 5      9493 , 4      2747   (n,7)

x_train , x_test , y_train , y_test = train_test_split(x,y ,test_size=0.3 , random_state= 2222 ,shuffle=True, stratify=y ) # 0

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier , XGBRegressor
from sklearn.ensemble import RandomForestClassifier , VotingClassifier

xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model = VotingClassifier( estimators= [ ('XGB', xgb) , ('RF',rf) ,('LR',lr) ] ,
                        #  voting='hard' ,# hard가 default
                         voting='soft'
                         )
#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)

# 원래 값
# model.score 0.10956145584725537

# False
# model.score 0.7258640077106664

# True
# model.score 0.7258295850927116

# soft
# model.score 0.899795759133468
# hard
# model.score 0.884414586010648