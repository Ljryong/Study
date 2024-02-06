import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input,Conv2D,Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical #
import matplotlib.pyplot as plt

path = "C:\\_data\\dacon\\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0 )

test_csv = pd.read_csv(path + "test.csv", index_col=0 )

submission_csv = pd.read_csv(path + "sample_submission.csv")


train_csv = train_csv[train_csv['주택소유상태'] != 'ANY']
test_csv.loc[test_csv['대출목적'] == '결혼' , '대출목적'] = '기타'


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder()



train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])
train_csv['대출목적'] = le.fit_transform(train_csv['대출목적'])
train_csv['대출기간'] = train_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])

test_csv['주택소유상태'] = le.fit_transform(test_csv['주택소유상태'])
test_csv['대출목적'] = le.fit_transform(test_csv['대출목적'])
test_csv['대출기간'] = test_csv['대출기간'].str.slice(start=0,stop=3).astype(int)
test_csv['근로기간'] = le.fit_transform(test_csv['근로기간'])

train_csv['대출등급'] = le.fit_transform(train_csv['대출등급'])

x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']




x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=66101,stratify=y,shuffle=True,)
# smote=SMOTE(random_state=29,k_neighbors=4)
# x_train,y_train=smote.fit_resample(x_train,y_train)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler 

scaler=StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape)



model = Sequential()
model.add(Dense(102 ,input_shape= (13,),activation='swish'))
model.add(Dense(15,activation= 'swish'))
model.add(Dense(132,activation= 'swish'))
model.add(Dense(13, activation= 'swish'))
model.add(Dense(64,activation= 'swish'))
model.add(Dense(7,activation='softmax'))


import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
MCP_path = "../_data/_save/MCP/"
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([MCP_path, 'k23_', date, '_', filename])

hist=model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss',mode='auto',patience= 3000,verbose=1,restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,filepath=filepath,)

model.fit(x_train, y_train, epochs=1000000, batch_size = 1500,validation_split=0.15, callbacks=[es, mcp],verbose=1 )

results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)  
y_submit = model.predict(test_csv)
submit = np.argmax(y_submit, axis=1)
submitssion = le.inverse_transform(submit)
      
submission_csv['대출등급'] = submitssion

f1 = f1_score(y_test, y_predict, average='macro')
acc = accuracy_score(y_test, y_predict)
print("로스 : ", results[0])  
print("acc : ", results[1])  
print("f1 : ", f1)  
submission_csv.to_csv(path + "submission_0202_1.csv", index=False)