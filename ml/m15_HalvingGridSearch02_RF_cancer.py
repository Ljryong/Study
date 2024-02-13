import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score , mean_squared_error , mean_squared_log_error , accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split , KFold, cross_val_score , cross_val_predict
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC , SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


#1 데이터                     ####################################      2진 분류        ###########################################
datasets = load_breast_cancer()
# print(datasets)     
print(datasets.DESCR) # datasets 에 있는 describt 만 뽑아줘
print(datasets.feature_names)       # 컬럼 명들이 나옴

x = datasets.data       # x,y 를 정하는 것은 print로 뽑고 data의 이름과 target 이름을 확인해서 적는 것
y = datasets.target
print(x.shape,y.shape)  # (569, 30) (569,)

x_train , x_test, y_train , y_test = train_test_split(x,y,random_state=123,stratify=y,test_size=0.3, shuffle=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler , MaxAbsScaler , RobustScaler
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#2 모델구성
from sklearn.model_selection import StratifiedKFold , GridSearchCV , RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
kfold = StratifiedKFold(n_splits= 3 , shuffle=True , random_state= 1234 )

parameters =[
    {'n_estimators' : [100,200] ,'max_depth':[6,10,12],'min_samples_leaf' : [3,10]},
    {'max_depth': [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3,5,7,10],'min_samples_split' : [2,3,5,10]},
    {'min_samples_split' : [2,3,5,10] },
    {'n_jobs' : [-1,2,4], 'min_samples_split' : [2,3,5,10]}
]
#2 모델

# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold ,
#                     verbose=1,
#                     refit=True,
#                     n_jobs= -1 )

model = HalvingGridSearchCV(RandomForestClassifier(), parameters, cv = kfold ,
                                verbose=1,
                                refit=True,
                                n_jobs= -1 ,
                                random_state=66, 
                                # n_iter= 10,                 # halbingirdsearch에서는 n_iter 안먹음
                                min_resources=10,
                                factor=3)


#3 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4 평가
y_predict = model.predict(x_test)
print('accuracy_score' , accuracy_score(y_test,y_predict))

print(accuracy_score(y_test,y_predict))

print('시간 : ' , round(end - start,2))

# hist = model.fit(x_train , y_train,epochs = 1000000 , batch_size = 1 ,  validation_split= 0.2 , callbacks=[es] ,)

#4 평가, 예측
# loss = model.evaluate(x_test,y_test)        # evaluate = predict로 훈련한 x_test 값을 y_test 값이랑 비교하여 평가한다.

# print(np.round(y_predict))                  # round = 반올림 시켜주는 것


# def ACC(aaa, bbb) :                                  # aaa,bbb 가 값이 들어가 있는 것이 아니라 '빈 박스' 같은 느낌이다.
#     return np.sqrt(mean_squared_error(aaa, bbb))     
# acc = ACC(y_test,y_predict)                          # 빈 박스를 여기 ACC() 로 묶어주고 빈 박스의 이름을 정해준 것이다.
# print("ACC : ", acc)


# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c = 'red' , label = 'loss' , marker='.')
# # c = 'red' , label = 'loss' , marker='.' // c = color , label = 이름 , marker = 1 epoch 당 . 을 찍어주세요
# plt.plot(hist.history['val_loss'], c = 'blue' , label = 'val_loss' , marker='.')

# plt.plot(hist.history['accuracy'], c = 'green' , label = 'accuracy' , marker = '.')

# plt.legend(loc='upper right') # 라벨을 오른쪽 위에 달아주세요
# plt.title('boston loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()

# plt.show()

# print("loss = ",loss)
# print('r2 = ', r2)


# GridSearchCV
# Fitting 3 folds for each of 60 candidates, totalling 180 fits
# accuracy_score 0.9766081871345029
# 0.9766081871345029
# 시간 :  2.64


# Fitting 3 folds for each of 10 candidates, totalling 30 fits
# accuracy_score 0.9707602339181286
# 0.9707602339181286
# 시간 :  3.25



# ==========================================

# iter: 0
# n_candidates: 60
# n_resources: 10
# Fitting 3 folds for each of 60 candidates, totalling 180 fits
# ----------
# iter: 1
# n_candidates: 20
# n_resources: 30
# Fitting 3 folds for each of 20 candidates, totalling 60 fits
# ----------
# iter: 2
# n_candidates: 7
# n_resources: 90
# Fitting 3 folds for each of 7 candidates, totalling 21 fits
# ----------
# iter: 3
# n_candidates: 3
# n_resources: 270
# Fitting 3 folds for each of 3 candidates, totalling 9 fits
# accuracy_score 0.9649122807017544
# 0.9649122807017544
