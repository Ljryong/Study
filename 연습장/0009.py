from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# dacon , keras 손코딩 1 번씩 

#1 
path = "c:/_data/dacon/wine//"

train_csv = pd.read_csv(path + "train.csv" , index_col= 0)      # index_col : 컬럼을 무시한다. //  index_col= 0 는 0번째 컬럼을 무시한다. 
test_csv = pd.read_csv(path + "test.csv" , index_col= 0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


print(train_csv)        # [5497 rows x 13 columns]
print(test_csv)         # [1000 rows x 12 columns]

######################## 사이킷런 문자데이터 수치화 ##################
from sklearn.preprocessing import LabelEncoder      # 문자데이터를 알파벳 순서대로 수치화한다
lab = LabelEncoder()
lab.fit(train_csv['type'])
trainlab_csv = lab.transform(train_csv['type'])

print(trainlab_csv)

lab = LabelEncoder()
lab.fit(test_csv['type'])
testlab_csv = lab.transform(test_csv['type'])
print(testlab_csv)

a = train_csv.drop['type'] + trainlab_csv
print(a)

#######################################################################
# github.com/stascorp/rdpwrap/releases
#######################################################################



######## keras에 있는 데이터 수치화 방법 ##########
# train_csv['type'] = train_csv['type'].replace({'white': 0, 'red':1})
# test_csv['type'] = test_csv['type'].replace({'white': 0, 'red':1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']
# print(train_csv)
# print(y.shape)          # (5497,1)

from keras.utils import to_categorical
one_hot_y = to_categorical(y)
print(one_hot_y.shape)  # (5497, 10)

# one_hot = pd.get_dummies(y)
# print(one_hot)          # [5497 rows x 2 columns]


x_train , x_test , y_train , y_test = train_test_split(x,one_hot_y, test_size=0.3 , random_state= 5 , shuffle=True , stratify= y )

es = EarlyStopping(monitor='val_loss' , mode = 'min', verbose=1, patience= 5s0 , restore_best_weights=True )


#2 
model = Sequential()
model.add(Dense(1024,input_dim = 12 ))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(10, activation= 'softmax'))

#3 
model.compile(loss = 'categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train,epochs=1000000 ,batch_size = 10 , validation_split=0.2 , callbacks=[es], verbose= 1)

#4
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
y_submit = model.predict(test_csv)

submission_csv['quality'] = np.argmax(y_submit, axis=1)
# y_submit도 결과값을 뽑아내야 되는데 그냥 뽑으면 소수점을 나와서 argmax로 위치값의 정수를 뽑아줘야한다.

submission_csv.to_csv(path + 'submission_0112.csv',index = False)

arg_test = np.argmax(y_test , axis = 1)
arg_predict = np.argmax(y_predict , axis = 1)

def ACC(arg_test,arg_predict):
    return accuracy_score(arg_test,arg_predict)
acc = ACC(arg_test,arg_predict)


print(loss)
print("Acc = ",acc)


# Epoch 46: early stopping
# 52/52 [==============================] - 0s 942us/step - loss: 1.1259 - acc: 0.5139
# 52/52 [==============================] - 0s 873us/step
# 32/32 [==============================] - 0s 1ms/step
# [1.12594735622406, 0.513939380645752]
# Acc =  0.5139393939393939


# 1
# Epoch 238: early stopping
# 52/52 [==============================] - 0s 868us/step - loss: 1.0999 - acc: 0.5364
# 52/52 [==============================] - 0s 456us/step
# 32/32 [==============================] - 0s 1ms/step
# [1.0999300479888916, 0.5363636612892151]
# Acc =  0.5363636363636364


# Epoch 168: early stopping
# 52/52 [==============================] - 0s 976us/step - loss: 1.1065 - acc: 0.5291
# 52/52 [==============================] - 0s 907us/step
# 32/32 [==============================] - 0s 1ms/step
# [1.1064668893814087, 0.5290908813476562]
# Acc =  0.5290909090909091


# Epoch 178: early stopping
# 52/52 [==============================] - 0s 670us/step - loss: 1.0823 - acc: 0.5339
# 52/52 [==============================] - 0s 749us/step
# 32/32 [==============================] - 0s 588us/step
# [1.0823452472686768, 0.5339394211769104]
# Acc =  0.5339393939393939



# Epoch 158: early stopping
# 52/52 [==============================] - 0s 975us/step - loss: 1.0833 - acc: 0.5352
# 52/52 [==============================] - 0s 1ms/step
# 32/32 [==============================] - 0s 1ms/step
# [1.0832772254943848, 0.5351515412330627]
# Acc =  0.5351515151515152







