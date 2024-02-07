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

from sklearn.linear_model import Perceptron , LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
models = [LinearSVC(),Perceptron(),LogisticRegression(),RandomForestClassifier(),DecisionTreeClassifier(),KNeighborsClassifier()]



#3 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy' , 'mse', 'mae'])     
# binary_crossentropy = 2진 분류 = y 가 2개면 무조건 이걸 사용한다
# accuracy = 정확도 = acc
# metrics로 훈련되는걸 볼 수는 있지만 가중치에 영향을 미치진 않는다.

# metrics가 accuracy 

############## 훈련 반복 for 문 ###################
for model in models :
    model.fit(x_train,y_train)
    result = model.score(x_test,y_test)
    print(f'{type(model).__name__} score ',result)
    y_predict = model.predict(x_test)
    print(f'{type(model).__name__} predict ',accuracy_score(y_test,y_predict))


# Epoch 35: early stopping
# 5447/5447 [==============================] - 7s 1ms/step - loss: 0.6882 - acc: 0.7010
# 5447/5447 [==============================] - 7s 1ms/step
# (174304,)
# (174304,)
# acc =  0.7009707178263265
# batch = 2500

# 0
# Epoch 22: early stopping
# 5447/5447 [==============================] - 7s 1ms/step - loss: 0.6842 - acc: 0.7016
# 5447/5447 [==============================] - 6s 1ms/step
# (174304,)
# (174304,)
# acc =  0.7015673765375436
# batch = 1000

# 2
# Epoch 22: early stopping
# 5447/5447 [==============================] - 6s 1ms/step - loss: 0.6818 - acc: 0.7041
# 5447/5447 [==============================] - 6s 1ms/step
# (174304,)
# (174304,)
# acc =  0.7041146502662016
# batch = 1000


# MinMaxScaler
# Epoch 93: early stopping
# 5447/5447 [==============================] - 5s 831us/step - loss: 0.6298 - acc: 0.7239
# 5447/5447 [==============================] - 4s 785us/step
# (174304,)
# (174304,)
# acc =  0.7239191297962181
# reslut =  [0.6298187375068665, 0.723919153213501]

# StandardScaler
# Epoch 58: early stopping
# 5447/5447 [==============================] - 4s 806us/step - loss: 0.6309 - acc: 0.7228
# 5447/5447 [==============================] - 4s 806us/step
# (174304,)
# (174304,)
# acc =  0.7227774463007159
# reslut =  [0.6309492588043213, 0.7227774262428284]


# MaxAbsScaler
# Epoch 108: early stopping
# 5447/5447 [==============================] - 5s 842us/step - loss: 0.6302 - acc: 0.7242
# 5447/5447 [==============================] - 4s 804us/step
# (174304,)
# (174304,)
# acc =  0.7242002478428493
# reslut =  [0.630240261554718, 0.7242002487182617]

# RobustScaler
# Epoch 51: early stopping
# 5447/5447 [==============================] - 9s 2ms/step - loss: 0.6287 - acc: 0.7247
# 5447/5447 [==============================] - 8s 1ms/step
# (174304,)
# (174304,)
# acc =  0.7246764273912245
# reslut =  [0.6287251114845276, 0.7246764302253723]


# acc =  0.6820210666421883
# reslut =  0.6820210666421883


# LinearSVC score  0.7131620616853314
# LinearSVC predict  0.7131620616853314
# Perceptron score  0.6213053056728475
# Perceptron predict  0.6213053056728475
# LogisticRegression score  0.7248829630989535
# LogisticRegression predict  0.7248829630989535
# RandomForestClassifier score  0.9530303378006242
# RandomForestClassifier predict  0.9530303378006242
# DecisionTreeClassifier score  0.9359796677069947
# DecisionTreeClassifier predict  0.9359796677069947
# KNeighborsClassifier score  0.9254979805397466
# KNeighborsClassifier predict  0.9254979805397466
