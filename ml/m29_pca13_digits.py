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

x_train, x_test , y_train , y_test = train_test_split(x,y,test_size= 0.3 , random_state= 123 , stratify=y , shuffle=True)


kfold = StratifiedKFold(n_splits= 3 , shuffle=True , random_state= 1234 )


#2 모델
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

#2 모델구성
from sklearn.ensemble import RandomForestRegressor

columns = datasets.feature_names
columns = x.columns
x = pd.DataFrame(x,columns=columns)
from sklearn.decomposition import PCA
for i in range(len(x.columns)) :
    pca = PCA(n_components=i+1)
    x_train_2 = pca.fit_transform(x_train)
    x_test_2 = pca.transform(x_test)
    model = RandomForestRegressor()
    model.fit(x_train_2,y_train)
    result = model.score(x_test_2,y_test)
    print('n_components = ', i+1 ,'result',result)
    print('='*50)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=6)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
#2
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor

# model = RandomForestRegressor()
model = RandomForestClassifier()

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
print(x.shape)

evr = pca.explained_variance_ratio_

evr_cumsum = np.cumsum(evr)   
print(evr_cumsum)


