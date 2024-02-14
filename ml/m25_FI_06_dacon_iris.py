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

# print(train_csv.shape)          # (120, 5)
# print(test_csv.shape)           # (30, 4)

print('======'*50)
x = train_csv.drop(['species'],axis=1)
y = train_csv['species']

# x = np.delete(x,0,axis=1)

''' 25퍼 미만 열 삭제 '''
# columns = datasets.feature_names
columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
fi_str = "0.00737395 0.02304103 0.59607338 0.37351164"
print('======'*50)
 
''' str에서 숫자로 변환하는 구간 '''
fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
print(fi_float)
fi_list = pd.Series(fi_float)
print('======'*50)

''' 25퍼 미만 인덱스 구하기 '''
low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print('low_idx_list',low_idx_list)
print('======'*50)

''' 25퍼 미만 제거하기 '''
low_col_list = [x.columns[index] for index in low_idx_list]
# 이건 혹여 중복되는 값들이 많아 25퍼이상으로 넘어갈시 25퍼로 자르기
if len(low_col_list) > len(x.columns) * 0.25:   
    low_col_list = low_col_list[:int(len(x.columns)*0.25)]
print('low_col_list',low_col_list)
x.drop(low_col_list,axis=1,inplace=True)
print("after x.shape",x.shape)

print('======'*50)

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
model = LinearSVC(C=100)

#3
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
# model.fit(x_train,y_train, epochs=1000000 , batch_size = 100 , verbose=1, callbacks=[es] , validation_split=0.2 )
model.fit(x_train,y_train)

#4
result = model.score(x_test,y_test)
y_predict = model.predict(x_test)
# y_submit = model.predict(test_csv)

from sklearn.svm import LinearSVR

from sklearn.linear_model import Perceptron , LogisticRegression , LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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
    
# DecisionTreeClassifier : [0.02381965 0.         0.37343673 0.60274361] 1.0
# DecisionTreeClassifier result 1.0
# RandomForestClassifier : [0.07506213 0.05931356 0.42045215 0.44517216] 1.0
# RandomForestClassifier result 1.0
# GradientBoostingClassifier : [0.00737395 0.02304103 0.59607338 0.37351164] 1.0
# GradientBoostingClassifier result 1.0
# XGBClassifier : [0.0102429  0.04045254 0.77684844 0.17245616] 0.9722222222222222
# XGBClassifier result 0.9722222222222222

# DecisionTreeClassifier : [0.02381965 0.37343673 0.60274361] 1.0
# DecisionTreeClassifier result 1.0
# RandomForestClassifier : [0.17719947 0.46540495 0.35739559] 1.0
# RandomForestClassifier result 1.0
# GradientBoostingClassifier : [0.02506157 0.57415453 0.40078391] 1.0
# GradientBoostingClassifier result 1.0
# XGBClassifier : [0.04364544 0.75705636 0.19929814] 0.9722222222222222
# XGBClassifier result 0.9722222222222222