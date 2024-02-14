from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

#1
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape,y.shape)      # (581012, 54) (581012,)
print(pd.value_counts(y))   # 2    283301 , 1    211840 , 3     35754 , 7     20510 , 6     17367 , 5      9493 , 4      2747   (n,7)

# one_hot = pd.get_dummies(y)

# from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(-1,1)
# ohe = OneHotEncoder()
# ohe.fit(y)
# one_hot = ohe.transform(y).toarray()

# print(one_hot)



x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state= 2 ,shuffle=True, stratify=y ) # 0

es= EarlyStopping(monitor='val_loss' , mode = 'min', verbose= 1 ,patience=10, restore_best_weights=True )


# print(datasets.DESCR)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

models = [DecisionTreeClassifier(random_state = 777), RandomForestClassifier(random_state = 777) , 
          GradientBoostingClassifier(random_state = 777),XGBClassifier()]

############## 훈련 반복 for 문 ###################a
for model in models :
    model.fit(x_train,y_train)
    result = model.score(x_test,y_test)
    print(type(model).__name__,':',model.feature_importances_ ,result)
   # y_predict = model.predict(x_test)
    print(type(model).__name__,'result',result)







