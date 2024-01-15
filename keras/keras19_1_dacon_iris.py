from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd


#1
path = "c:/_data/dacon/iris//"

train_csv = pd.read_csv(path + 'train.csv' , index_col = 0)
test_csv = pd.read_csv(path + 'test.csv' , index_col = 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

print(train_csv.shape)          # (120, 5)
print(test_csv.shape)           # (30, 4)


x = train_csv.drop(['species'],axis=1)
y = train_csv['species']

one_hot = pd.get_dummies(y)

print(one_hot)      # [120 rows x 3 columns]
print(one_hot.shape)    # (120, 3)

x_train , x_test , y_train , y_test = train_test_split(x,one_hot,test_size=0.3, random_state= 200 ,shuffle=True , stratify = y )
# OneHot 을 했으면 y가 아니라 OneHot을 넣어줘야한다.
es = EarlyStopping(monitor='val_loss', mode='min' , verbose= 1 , restore_best_weights=True , patience= 1000  )

#2
model = Sequential()
model.add(Dense(64,input_dim = 4))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(3, activation= 'softmax'))

#3
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['acc'])
model.fit(x_train,y_train, epochs=1000000 , batch_size = 100 , verbose=1, callbacks=[es] , validation_split=0.2 )

#4
result = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)

submission_csv['species'] = np.argmax(y_submit, axis=1)
# y_submit도 결과값을 뽑아내야 되는데 그냥 뽑으면 소수점을 나와서 argmax로 위치값의 정수를 뽑아줘야한다.



submission_csv.to_csv(path + 'submission_0112.csv',index = False)




arg_test = np.argmax(y_test , axis = 1)
arg_predict = np.argmax(y_predict , axis = 1)

def ACC(arg_test,arg_predict):
    return accuracy_score(arg_test,arg_predict)
acc = ACC(arg_test,arg_predict)


print(result)
print("Acc = ",acc)




# 선의형의 1점



