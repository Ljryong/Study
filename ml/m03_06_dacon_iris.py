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

print(train_csv.shape)          # (120, 5)
print(test_csv.shape)           # (30, 4)


x = train_csv.drop(['species'],axis=1)
y = train_csv['species']




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
models = [LinearSVC(),Perceptron(),LogisticRegression(),RandomForestClassifier(),DecisionTreeClassifier(),KNeighborsClassifier()]

############## 훈련 반복 for 문 ###################
for model in models :
    model.fit(x_train,y_train)
    result = model.score(x_test,y_test)
    print(f'{type(model).__name__} score ',result)
    y_predict = model.predict(x_test)
    print(f'{type(model).__name__} predict ',accuracy_score(y_test,y_predict))



# submission_csv['species'] = np.argmax(y_submit, axis=1)
# y_submit도 결과값을 뽑아내야 되는데 그냥 뽑으면 소수점을 나와서 argmax로 위치값의 정수를 뽑아줘야한다.



# submission_csv.to_csv(path + 'submission_0112.csv',index = False)




# arg_test = np.argmax(y_test , axis = 1)
# arg_predict = np.argmax(y_predict , axis = 1)

# def ACC(arg_test,arg_predict):
#     return accuracy_score(arg_test,arg_predict)
# acc = ACC(arg_test,arg_predict)


print(result)
print("Acc = ",accuracy_score(y_test,y_predict))

# 0.9444444444444444
# Acc =  0.9444444444444444


# 선의형의 1점


# LinearSVC score  0.9722222222222222
# LinearSVC predict  0.9722222222222222
# Perceptron score  0.8333333333333334
# Perceptron predict  0.8333333333333334
# LogisticRegression score  1.0
# LogisticRegression predict  1.0
# RandomForestClassifier score  1.0
# RandomForestClassifier predict  1.0
# DecisionTreeClassifier score  1.0
# DecisionTreeClassifier predict  1.0
# KNeighborsClassifier score  1.0
# KNeighborsClassifier predict  1.0
# 1.0
# Acc =  1.0
