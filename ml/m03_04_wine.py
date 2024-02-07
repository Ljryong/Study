from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC


#1

datasets = load_wine()
x= datasets.data
y= datasets.target

print(x.shape,y.shape)      # (178, 13) (178,)
print(pd.value_counts(y))   # 1    71 , 0    59 , 2    48


x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.3, random_state= 0 ,shuffle=True, stratify = y)

es = EarlyStopping(monitor='val_loss', mode = 'min' , verbose= 1 ,patience=20 ,restore_best_weights=True)

#2
from sklearn.svm import LinearSVR

from sklearn.linear_model import Perceptron , LogisticRegression , LinearRegression
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



# 결과 0.7037037037037037
# [0 1 0 0 1 2 1 2 1 2 0 1 2 0 2 1 1 1 2 1 0 2 1 1 1 1 1 2 2 1 1 2 1 1 1 1 1
#  1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1]
# accuracy :  0.7037037037037037


# LinearSVC score  0.8518518518518519
# LinearSVC predict  0.8518518518518519
# Perceptron score  0.5370370370370371
# Perceptron predict  0.5370370370370371
# LogisticRegression score  0.9444444444444444
# LogisticRegression predict  0.9444444444444444
# RandomForestClassifier score  1.0
# RandomForestClassifier predict  1.0
# DecisionTreeClassifier score  0.9629629629629629
# DecisionTreeClassifier predict  0.9629629629629629
# KNeighborsClassifier score  0.7222222222222222
# KNeighborsClassifier predict  0.7222222222222222