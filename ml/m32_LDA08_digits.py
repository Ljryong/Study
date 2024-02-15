from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense 
import time
from sklearn.model_selection import train_test_split , RandomizedSearchCV , GridSearchCV , StratifiedKFold , cross_val_predict , cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# import numpy as np

datasets = load_digits()
x = datasets.data
y = datasets.target

print(np.unique(y))

x_train, x_test , y_train , y_test = train_test_split(x,y,test_size= 0.3 , random_state= 123 , stratify=y , shuffle=True)


kfold = StratifiedKFold(n_splits= 3 , shuffle=True , random_state= 1234 )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
lda = LinearDiscriminantAnalysis(n_components=9)
x_train2 = lda.fit_transform(x_train,y_train)
x_test2 = lda.transform(x_test)
#2
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor

# model = RandomForestRegressor()
model = RandomForestClassifier()

#3 훈련
model.fit(x_train2,y_train)

#4 평가,예측
result = model.score(x_test2,y_test)
print('model.score' , result)
print(x.shape)

evr = lda.explained_variance_ratio_

evr_cumsum = np.cumsum(evr)   
print(evr_cumsum)


# model.score 0.9518518518518518
# (1797, 64)
# [0.28060193 0.46816127 0.63866271 0.75642415 0.84340146 0.90838518
#  0.95120279 0.98085574 1.        ]