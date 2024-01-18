from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from keras.models import Sequential , Model
from keras.layers import Dense , Dropout , Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping , ModelCheckpoint
import datetime
import time

#1

datasets = load_wine()
x= datasets.data
y= datasets.target

print(x.shape,y.shape)      # (178, 13) (178,)
print(pd.value_counts(y))   # 1    71 , 0    59 , 2    48


from keras.utils import to_categorical
one_hot = to_categorical(y)             # 행렬 데이터로 바꿔주는 것
print(one_hot)
print(one_hot.shape)                    # (178, 3)


one_hot = pd.get_dummies(y)


from sklearn.preprocessing import OneHotEncoder
y = y.reshape(-1,1)
ohe = OneHotEncoder()

ohe.fit(y)
y_ohe = ohe.transform(y).toarray()

print(y_ohe)
print(y_ohe.shape)     # (178, 3)



x_train , x_test , y_train , y_test = train_test_split(x,y_ohe, test_size = 0.3, random_state= 0 ,shuffle=True, stratify = y)

es = EarlyStopping(monitor='val_loss', mode = 'min' , verbose= 1 ,patience= 10 ,restore_best_weights=True)

date = datetime.datetime.now()
date = date.strftime('%m%d-%H%M')
path = 'c:/_data/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path , 'k28_8_', date , '_', filename ])

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


#2
# model = Sequential()
# model.add(Dense(64,input_shape = 13))
# model.add(Dense(128))
# model.add(Dropout(0.3))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(3,activation='softmax'))

#2-1
input = Input(shape = (13,))
d1 = Dense(64)(input)
d2 = Dense(128)(d1)
drop1 = Dropout(0.3)(d2)
d3 = Dense(64)(drop1)
d4 = Dense(32)(d3)
d5 = Dense(16)(d4)
output = Dense(3,activation='softmax')(d5)
model = Model(inputs = input , outputs = output)


#3
model.compile(loss="categorical_crossentropy", optimizer='adam' , metrics = ['acc'])
mcp = ModelCheckpoint(monitor='val_loss', mode='min' , verbose=1, save_best_only=True , filepath= filepath   )
start_time = time.time()

model.fit(x_train,y_train,epochs = 1000 , batch_size = 1 , verbose = 1 , callbacks=[mcp],validation_split=0.2)
end_time = time.time()

#4
result = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)


y_test = np.argmax(y_test,axis = 1)
y_predict = np.argmax(y_predict, axis = 1)
print(y_test.shape,y_predict.shape)

acc = accuracy_score(y_test,y_predict)

print('결과',result[0])
print('acc',result[1])
print(y_predict)
print("accuracy : ",acc)
print('시간 :' , end_time - start_time)


# 결과 0.19868147373199463

# MinMaxScaler
# 결과 2.9090735552017577e-05

# StandardScaler
# 결과 0.03268396109342575

# MaxAbsScaler
# 결과 0.07153265178203583

# RobustScaler
# 결과 0.003748755669221282

# Dropout
# 결과 0.3985663652420044


# cpu
# 시간 : 86.41531133651733
# gpu
# 시간 : 260.83926463127136
