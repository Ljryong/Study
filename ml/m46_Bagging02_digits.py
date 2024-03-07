from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense 
import time
from sklearn.model_selection import train_test_split , RandomizedSearchCV , GridSearchCV , StratifiedKFold , cross_val_predict , cross_val_score
from sklearn.ensemble import RandomForestClassifier , BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

datasets = load_digits()
x = datasets.data
y = datasets.target

x_train, x_test , y_train , y_test = train_test_split(x,y,test_size= 0.3 , random_state= 2222 , stratify=y , shuffle=True)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from xgboost import XGBClassifier, XGBRFClassifier

model = BaggingClassifier(LogisticRegression(),
                          n_jobs = -1,
                          n_estimators=10,
                        #   bootstrap=True,
                          bootstrap=False,
                          )
                          

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)

# 원래 값
# model.score 0.9518518518518518
# True
# model.score 0.9703703703703703
# False
# model.score 0.9703703703703703