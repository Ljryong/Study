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
# model = Sequential()
# model.add(Dense(64,input_dim = 13))
# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(3,activation='softmax'))
model = LinearSVC(C=150)

#3
# model.compile(loss="categorical_crossentropy", optimizer='adam' , metrics = ['acc'])
# model.fit(x_train,y_train,epochs = 1000000 , batch_size = 1 , verbose = 1 , callbacks=[es],validation_split=0.2)
model.fit(x_train , y_train)

#4
# result = model.evaluate(x_test,y_test)
result = model.score(x_test,y_test)
y_predict = model.predict(x_test)

acc = accuracy_score(y_test,y_predict)

print('결과',result)
# print('acc',result[1])
print(y_predict)
print("accuracy : ",acc)



# 결과 0.7037037037037037
# [0 1 0 0 1 2 1 2 1 2 0 1 2 0 2 1 1 1 2 1 0 2 1 1 1 1 1 2 2 1 1 2 1 1 1 1 1
#  1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1]
# accuracy :  0.7037037037037037


