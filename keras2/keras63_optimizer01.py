# restore_best_weights
# save_best_only
# 에 대한 고찰



from sklearn.datasets import load_boston , load_breast_cancer
import numpy as np
from keras.models import Sequential , load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
import pandas as pd

# warning 뜨는것을 없애는 방법, 하지만 아직 왜 뜨는지 모르니 보는것을 추천
import warnings
warnings.filterwarnings('ignore') 

# 현재 사이킷런 버전 1.3.0 보스턴 안됨, 그래서 삭제
# pip uninstall scikit-learn
# pip uninstall scikit-image
# pip uninstall scikit-learn-intelex

# pip install scikit-learn==0.23.2
datasets = load_breast_cancer()

print(datasets)
x = datasets.data
y = datasets.target
print(x)
print(x.shape)      #(506, 13)
print(y)
print(y.shape)      #(506,)

############################
# df = pd.DataFrame(x)
# Nan_num = df.isna().sum()
# print(Nan_num)
############################

print(datasets.feature_names)
# 'CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

print(datasets.DESCR)               
#[실습]
# train , test의 비율을 0.7 이상 0.9 이하
# R2 0.62 이상
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 51, stratify=y )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2
model = Sequential()
model.add(Dense(20,input_dim = 30))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(14))
model.add(Dense(7))
model.add(Dense(1,activation='sigmoid'))
model.summary()

# 3 컴파일, 훈련
from keras.optimizers import Adam
learning_rate = 1.0

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate) )

import datetime
date= datetime.datetime.now()       
print(date)                         
date = date.strftime('%m%d_%H%M')   
print(date)                         
path ='../_data/_save/MCP/'     
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'        

filepath = ''.join([path ,'k25_', date , '_' , filename])        # join = 이어주는것 


hist = model.fit(x_train,y_train, epochs = 200 , batch_size= 32 , validation_split=0.2 , verbose = 1 )




# 4 평가, 예측
print('============================== 1. 기본 출력 ===============================')

loss = model.evaluate(x_test,y_test,verbose=0)

y_predict = model.predict(x_test,verbose=0)
print('lr : {0}, 로스 : {1} '.format(learning_rate,loss))
acc = accuracy_score(y_test,np.round(y_predict))
print('lr : {0} , ACC : {1} '.format(learning_rate,acc))

print('==============================')
print(hist.history['val_loss'])
print('==============================')

""" 100번
# lr : 1.0, 로스 : 244173438976.0
# lr : 1.0 , ACC : 0.9649122807017544

# lr : 0.1, 로스 : 253.0746307373047 
# lr : 0.1 , ACC : 0.9649122807017544

# lr : 0.01, 로스 : 0.1847223937511444 
# lr : 0.01 , ACC : 0.9590643274853801

# lr : 0.001, 로스 : 0.11361335963010788 
# lr : 0.001 , ACC : 0.9590643274853801

# lr : 0.0001, 로스 : 0.06892313808202744 
# lr : 0.0001 , ACC : 0.9707602339181286 """

""" 200번
lr : 1.0, 로스 : 4590491136.0 
lr : 1.0 , ACC : 0.9532163742690059

lr : 0.1, 로스 : 370.5730895996094 
lr : 0.1 , ACC : 0.9649122807017544

lr : 0.01, 로스 : 0.21450725197792053 
lr : 0.01 , ACC : 0.9649122807017544

lr : 0.001, 로스 : 0.1644132137298584 
lr : 0.001 , ACC : 0.9590643274853801

lr : 0.0001, 로스 : 0.0847698375582695 
lr : 0.0001 , ACC : 0.9707602339181286 """