import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import catboost as cb
import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
import random

#1 데이터
path = 'c:/_data/kaggle/fat//'

train_csv = pd.read_csv(path + 'train.csv',index_col=0)
test_csv = pd.read_csv(path + 'test.csv',index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

print(train_csv.isna().sum())
print(test_csv.isna().sum())

le = LabelEncoder()
le.fit(train_csv['Gender'])
train_csv['Gender'] = le.transform(train_csv['Gender'])
test_csv['Gender'] = le.transform(test_csv['Gender'])

le.fit(train_csv['family_history_with_overweight'])
train_csv['family_history_with_overweight'] = le.transform(train_csv['family_history_with_overweight'])
test_csv['family_history_with_overweight'] = le.transform(test_csv['family_history_with_overweight'])

le.fit(train_csv['FAVC'])
train_csv['FAVC'] = le.transform(train_csv['FAVC'])
test_csv['FAVC'] = le.transform(test_csv['FAVC'])

le.fit(train_csv['SMOKE'])
train_csv['SMOKE'] = le.transform(train_csv['SMOKE'])
test_csv['SMOKE'] = le.transform(test_csv['SMOKE'])

le.fit(train_csv['SCC'])
train_csv['SCC'] = le.transform(train_csv['SCC'])
test_csv['SCC'] = le.transform(test_csv['SCC'])

le.fit(train_csv['NObeyesdad'])
train_csv['NObeyesdad'] = le.transform(train_csv['NObeyesdad'])

train_csv['CAEC'] = train_csv['CAEC'].replace({'Always': 0 , 'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })
test_csv['CAEC'] = test_csv['CAEC'].replace({'Always': 0 , 'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })

train_csv['CALC'] = train_csv['CALC'].replace({'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })
test_csv['CALC'] = test_csv['CALC'].replace({'Always' : 2 , 'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })

train_csv['MTRANS'] = train_csv['MTRANS'].replace({'Automobile': 0 , 'Bike' : 1, 'Motorbike' : 2, 'Public_Transportation' : 3,'Walking' : 4})
test_csv['MTRANS'] = test_csv['MTRANS'].replace({'Automobile': 0 , 'Bike' : 1, 'Motorbike' : 2, 'Public_Transportation' : 3,'Walking' : 4})

x = train_csv.drop(['NObeyesdad'], axis= 1)
y = train_csv['NObeyesdad']

from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , RobustScaler

print(x.shape)

x_train , x_test , y_train , y_test = train_test_split(x,y, random_state=123 , test_size=0.3 , shuffle=True , stratify=y )

# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

es = EarlyStopping(monitor='val_loss', mode = 'min' , patience= 30 , restore_best_weights=True , verbose= 1 )

rlr = ReduceLROnPlateau(monitor='val_loss' , mode='auto' , patience= 20 , verbose= 1 , 
                        factor=0.5          # 갱신이 없으면 learning rate를 내가 지정한 수치(0.5) 만큼 나눈다
                                            # learning rate 의 default = 0.001
                                            # 이걸 쓰려면 default 보다 높게 잡고 많이 내려간 뒤 낮아지는게 좋음
                        )

#2 모델구성

#3 훈련
from keras.optimizers import Adam
learning_rates = [ 1.0, 0.1, 0.01, 0.001, 0.0001 ]
# learning_rate = 0.01
for learning_rate in learning_rates :
    
    model = Sequential()
    model.add(Dense(128 , input_shape =(16,) ))
    model.add(Dense(32,activation = 'relu' ))
    model.add(Dense(64,activation = 'relu' ))
    model.add(Dense(32,activation = 'relu' ))
    model.add(Dense(64,activation = 'relu' ))
    model.add(Dense(32,activation = 'relu' ))
    model.add(Dense(7,activation = 'softmax' ))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate) , metrics=['acc'])
    model.fit(x_train,y_train, epochs = 100 , batch_size= 700 , validation_split=0.15 , callbacks = [es, rlr] , verbose= 0 )

    #4
    loss = model.evaluate(x_test,y_test)
    y_predict = model.predict(x_test)
    arg_pre = np.argmax(y_predict,axis=1)
    # arg_test = np.argmax(y_test,axis=1)

    y_predict = model.predict(x_test,verbose=0)
    print('lr : {0}, 로스 : {1} '.format(learning_rate,loss[0]))
    acc = accuracy_score(y_test,arg_pre)
    print('lr : {0} , ACC : {1} '.format(learning_rate,acc))



""" 
lr : 1.0, 로스 : 1.9309966564178467 
lr : 1.0 , ACC : 0.19492614001284522

lr : 0.1, 로스 : 1.9309335947036743 
lr : 0.1 , ACC : 0.19492614001284522

lr : 0.01, 로스 : 0.37707847356796265 
lr : 0.01 , ACC : 0.8720295439948619

lr : 0.001, 로스 : 0.3743060231208801 
lr : 0.001 , ACC : 0.8717084136159281

lr : 0.0001, 로스 : 0.43396812677383423 
lr : 0.0001 , ACC : 0.8461785484906872
"""
