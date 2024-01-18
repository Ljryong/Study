from keras.models import Sequential , Model
from keras.layers import Dense , Dropout , Input
from keras.callbacks import EarlyStopping , ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import datetime
import time

#1 
path = "c:/_data/dacon/wine//"

train_csv = pd.read_csv(path + "train.csv" , index_col= 0)      # index_col : 컬럼을 무시한다. //  index_col= 0 는 0번째 컬럼을 무시한다. 
test_csv = pd.read_csv(path + "test.csv" , index_col= 0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


# print(train_csv)        # [5497 rows x 13 columns]
# print(test_csv)         # [1000 rows x 12 columns]

# ######################## 사이킷런 문자데이터 수치화 ##################
# from sklearn.preprocessing import LabelEncoder      # 문자데이터를 알파벳 순서대로 수치화한다
# lab = LabelEncoder()
# lab.fit(train_csv)
# trainlab_csv = lab.transform(train_csv)
# print(trainlab_csv)


# #####################################################################

####### keras에 있는 데이터 수치화 방법 ##########
train_csv['type'] = train_csv['type'].replace({'white': 0, 'red':1})
test_csv['type'] = test_csv['type'].replace({'white': 0, 'red':1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']
# print(train_csv)
# print(y.shape)          # (5497,1)

from keras.utils import to_categorical
one_hot_y = to_categorical(y)
print("+", one_hot_y.shape)
one_hot_y = np.delete(one_hot_y, 0, axis=1)
one_hot_y = np.delete(one_hot_y, 0, axis=1)
one_hot_y = np.delete(one_hot_y, 0, axis=1)
print("-", one_hot_y.shape)
print(one_hot_y.shape)  # (5497, 10)

# one_hot = pd.get_dummies(y)
# print(one_hot)          # [5497 rows x 2 columns]


x_train , x_test , y_train , y_test = train_test_split(x,one_hot_y, test_size=0.3 , random_state= 971 , shuffle=True , stratify= y )

es = EarlyStopping(monitor='val_loss' , mode = 'min', verbose=1, patience= 10 , restore_best_weights=True )

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

date = datetime.datetime.now()
date = date.strftime('%m%d-%H%M')
path = 'c:/_data/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path , 'k26_10_', date , '_', filename ])

mcp = ModelCheckpoint(monitor='val_loss', mode='min' , verbose=1, save_best_only=True , filepath=  filepath   )

#2 
# model = Sequential()
# model.add(Dense(2048,input_shape = (12,) ))
# model.add(Dense(1024))
# model.add(Dropout(0.7))
# model.add(Dense(512))
# model.add(Dropout(0.1))
# model.add(Dense(256))
# model.add(Dropout(0.5))
# model.add(Dense(128))
# model.add(Dropout(0.4))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(7, activation= 'softmax'))

#2-1
input = Input(shape=(12,))
d1 = Dense(2048)(input)
d2 = Dense(1024)(d1)
drop1 = Dropout(0.7)(d2)
d3 = Dense(512)(drop1)
drop2 = Dropout(0.1)(d3)
d4 = Dense(256)(drop2)
drop3 = Dropout(0.5)(d4)
d5 = Dense(128)(drop3)
drop4 = Dropout(0.4)(d5)
d7 = Dense(64)(drop4)
d8 = Dense(32)(d7)
d9 = Dense(16)(d8)
output = Dense(7,activation='softmax')(d9)
model = Model(inputs = input , outputs = output)



#3 
model.compile(loss = 'categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
start_time = time.time()

model.fit(x_train,y_train,epochs=1000 ,batch_size = 10000 , validation_split=0.2 , callbacks=[mcp], verbose= 1)
end_time = time.time()

#4
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
# print(y_predict.shape)
y_submit = model.predict(test_csv)

submission_csv['quality'] = np.argmax(y_submit, axis=1)+3       # +3 밖에 써줘야 된다. argmax 전에 쓰면 위치값이 안뽑혀있는 상태이고, 안에 쓰면 소용이 없다.
# y_submit도 결과값을 뽑아내야 되는데 그냥 뽑으면 소수점을 나와서 argmax로 위치값의 정수를 뽑아줘야한다.

submission_csv.to_csv(path + 'submission_0112.csv',index = False)

arg_test = np.argmax(y_test , axis = 1)
arg_predict = np.argmax(y_predict , axis = 1)

def ACC(arg_test,arg_predict):
    return accuracy_score(arg_test,arg_predict)
acc = ACC(arg_test,arg_predict)


print(loss)
print("Acc = ",acc) 
print('시간 :' , end_time - start_time)


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

# Epoch 123: early stopping
# 52/52 [==============================] - 0s 914us/step - loss: 1.0967 - acc: 0.5309
# 52/52 [==============================] - 0s 938us/step
# 32/32 [==============================] - 0s 1ms/step
# [1.0966628789901733, 0.5309090614318848]
# Acc =  0.5309090909090909



# Epoch 68: early stopping
# 52/52 [==============================] - 0s 1ms/step - loss: 1.1319 - acc: 0.5327
# 52/52 [==============================] - 0s 1ms/step
# (1650, 7)
# 32/32 [==============================] - 0s 1ms/step
# [1.131880283355713, 0.5327273011207581]
# Acc =  0.5327272727272727
# PS C:\Study> [1.131880283355713, 0.5327273011207581]
# >> Acc =  0.5327272727272727

# Epoch 1718: early stopping
# 52/52 [==============================] - 0s 1ms/step - loss: 1.0814 - acc: 0.5448
# 52/52 [==============================] - 0s 1ms/step
# (1650, 7)
# 32/32 [==============================] - 0s 1ms/step
# [1.0814285278320312, 0.5448485016822815]
# Acc =  0.5448484848484848



# Epoch 9114: early stopping
# 52/52 [==============================] - 0s 1ms/step - loss: 1.0885 - acc: 0.5424
# 52/52 [==============================] - 0s 1ms/step
# (1650, 7)
# 32/32 [==============================] - 0s 1ms/step
# [1.0885460376739502, 0.5424242615699768]
# Acc =  0.5424242424242425


# MinMaxScaler
# Epoch 155: early stopping
# 52/52 [==============================] - 0s 1ms/step - loss: 1.0702 - acc: 0.5376
# 52/52 [==============================] - 0s 1ms/step
# 32/32 [==============================] - 0s 1ms/step
# [1.0702472925186157, 0.5375757813453674]
# Acc =  0.5375757575757576

# StandardScaler
# Epoch 135: early stopping
# 52/52 [==============================] - 0s 2ms/step - loss: 1.0620 - acc: 0.5412
# 52/52 [==============================] - 0s 1ms/step
# 32/32 [==============================] - 0s 1ms/step
# [1.062016248703003, 0.5412121415138245]
# Acc =  0.5412121212121213

# MaxAbsScaler
# Epoch 149: early stopping
# 52/52 [==============================] - 0s 2ms/step - loss: 1.0740 - acc: 0.5345
# 52/52 [==============================] - 0s 1ms/step
# 32/32 [==============================] - 0s 1ms/step
# [1.0740325450897217, 0.5345454812049866]
# Acc =  0.5345454545454545

# RobustScaler
# Epoch 140: early stopping
# 52/52 [==============================] - 0s 2ms/step - loss: 1.0639 - acc: 0.5345
# 52/52 [==============================] - 0s 1ms/step
# 32/32 [==============================] - 0s 2ms/step
# [1.0638880729675293, 0.5345454812049866]
# Acc =  0.5345454545454545


# MinMaxScaler
# Epoch 155: early stopping
# 52/52 [==============================] - 0s 1ms/step - loss: 1.0702 - acc: 0.5376
# 52/52 [==============================] - 0s 1ms/step
# 32/32 [==============================] - 0s 1ms/step
# [1.0702472925186157, 0.5375757813453674]
# Acc =  0.5375757575757576

# StandardScaler
# Epoch 135: early stopping
# 52/52 [==============================] - 0s 2ms/step - loss: 1.0620 - acc: 0.5412
# 52/52 [==============================] - 0s 1ms/step
# 32/32 [==============================] - 0s 1ms/step
# [1.062016248703003, 0.5412121415138245]
# Acc =  0.5412121212121213

# MaxAbsScaler
# Epoch 149: early stopping
# 52/52 [==============================] - 0s 2ms/step - loss: 1.0740 - acc: 0.5345
# 52/52 [==============================] - 0s 1ms/step
# 32/32 [==============================] - 0s 1ms/step
# [1.0740325450897217, 0.5345454812049866]
# Acc =  0.5345454545454545

# RobustScaler
# Epoch 140: early stopping
# 52/52 [==============================] - 0s 2ms/step - loss: 1.0639 - acc: 0.5345
# 52/52 [==============================] - 0s 1ms/step
# 32/32 [==============================] - 0s 2ms/step
# [1.0638880729675293, 0.5345454812049866]
# Acc =  0.5345454545454545




# Dropout
# Epoch 18: val_loss did not improve from 1.13189
# 308/308 [==============================] - 3s 11ms/step - loss: 1.1878 - acc: 0.5041 - val_loss: 1.1840 - val_acc: 0.5013
# Epoch 18: early stopping
# 52/52 [==============================] - 0s 3ms/step - loss: 1.0957 - acc: 0.5267
# 52/52 [==============================] - 0s 2ms/step
# 32/32 [==============================] - 0s 2ms/step
# [1.095739483833313, 0.5266666412353516]
# Acc =  0.5266666666666666


# cpu
# 시간 : 87.72841811180115
# gpu
# 시간 : 25.953370094299316

