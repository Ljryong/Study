from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping 
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder , LabelEncoder , MinMaxScaler

#1
path = 'c:/_data/dacon/dechul//'
train_csv = pd.read_csv(path+ 'train.csv' , index_col= 0)
test_csv = pd.read_csv(path+ 'test.csv' , index_col= 0)
submission_csv = pd.read_csv(path+ 'sample_submission.csv')


# print(train_csv)        # [96294 rows x 14 columns]
# print(test_csv)         # [64197 rows x 13 columns]



encoder = LabelEncoder()
encoder.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = encoder.transform(train_csv['주택소유상태'])
# encoder.fit(train_csv['대출목적'])
# train_csv['대출목적'] = encoder.transform(train_csv['대출목적'])
# encoder.fit(train_csv['대출기간'])
# train_csv['대출기간'] = encoder.transform(train_csv['대출기간'])
# encoder.fit(train_csv['근로기간'])
# train_csv['근로기간'] = encoder.transform(train_csv['근로기간'])



encoder.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])
# encoder.fit(test_csv['대출목적'])
# test_csv['대출목적'] = encoder.transform(test_csv['대출목적'])
# encoder.fit(test_csv['대출기간'])
# test_csv['대출기간'] = encoder.transform(test_csv['대출기간'])
# encoder.fit(test_csv['근로기간'])
# test_csv['근로기간'] = encoder.transform(test_csv['근로기간'])

encoder.fit(train_csv['대출등급'])
train_csv['대출등급'] = encoder.transform(train_csv['대출등급'])




# print(train_csv.dtypes)
# print(test_csv.dtypes)

# encoder.fit(test_csv['대출목적'])
# encoder.fit(train_csv['대출목적'])
# encoder.fit(train_csv['대출기간'])
# encoder.fit(test_csv['대출기간'])
# encoder.fit(train_csv['근로기간'])
# encoder.fit(test_csv['근로기간'])





train_csv['주택소유상태'] = train_csv['주택소유상태'].replace({'MORTGAGE' : 0 , 'OWN' : 1 , 'RENT': 2 , 'ANY' : 0}).astype(float)
test_csv['주택소유상태'] = test_csv['주택소유상태'].replace({'MORTGAGE' : 0 , 'OWN' : 1 , 'RENT': 2}).astype(float)

train_csv['대출목적'] = train_csv['대출목적'].replace({'부채 통합' : 0 , '주택 개선' : 2 , '주요 구매': 4 , '휴가' : 9  
                                                     , '의료' : 5 , '자동차' : 6 , '신용 카드' : 1 , '기타' : 3 , '주택개선' : 8,
                                                      '소규모 사업' : 7 , '이사' :  8 , '주택': 10 , '재생 에너지' : 11 })
test_csv['대출목적'] = test_csv['대출목적'].replace({'부채 통합' : 0 , '주택 개선' : 2 , '주요 구매': 4 , '휴가' : 9 ,
                                             '의료' : 5 , '자동차' : 6 , '신용 카드' : 1 , '기타' : 3 , '주택개선' : 8,
                                             '소규모 사업' : 7 , '이사' :  8 , '주택': 10 , '재생 에너지' : 11 , 
                                             '결혼' : 0 })

# 결혼은 train에 없는 라벨이다. 그래서 12 로 두든 2로 두든 아니면 없애든 값이 좋은걸로 비교해보면 된다.
train_csv['대출기간'] = train_csv['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)
test_csv['대출기간'] = test_csv['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)

train_csv['근로기간'] = train_csv['근로기간'].replace({'10+ years' : 10 ,            # 10으로 둘지 그 이상으로 둘지 고민
                                                      '2 years' : 2 , '< 1 year' : 0.7 , '3 years' : 3.5 , '1 year' : 1 ,
                                                      'Unknown' : 0 , '5 years' : 5 , '4 years' : 4 , '8 years' : 8 ,
                                                      '6 years' : 6 , '7 years' : 7 , '9 years' : 9 , '10+years' : 11,
                                                      '<1 year' : 0.5 , '3' : 3 , '1 years' : 1.5 })

test_csv['근로기간'] = test_csv['근로기간'].replace({'10+ years' : 10 ,            # 10으로 둘지 그 이상으로 둘지 고민
                                                      '2 years' : 2 , '< 1 year' : 0.7 , '3 years' : 3.5 , '1 year' : 1 ,
                                                      'Unknown' : 0 , '5 years' : 5 , '4 years' : 4 , '8 years' : 8 ,
                                                      '6 years' : 6 , '7 years' : 7 , '9 years' : 9 , '10+years' : 11,
                                                      '<1 year' : 0.5 , '3' : 3 , '1 years' : 1.5 })


# print(train_csv['대출기간'])

# print(pd.value_counts(test_csv['근로기간']))       # pd.value_counts() = 컬럼의 이름과 수를 알 수 있다.

x = train_csv.drop(['대출등급'],axis = 1 )
y = train_csv['대출등급']
# print(train_csv.dtypes)

# print(y.shape)      # (96294,)


# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)

# test_csv = scaler.transform(test_csv)       # test_csv도 같이 학습 시켜줘야 값이 나온다. 안해주면 소용이 없다.

y = y.values.reshape(-1,1)       # (96294, 1)

# print(y.shape) 
ohe = OneHotEncoder(sparse = False)
ohe.fit(y)
y_ohe = ohe.transform(y) 


x_train ,x_test , y_train , y_test = train_test_split(x,y_ohe,test_size = 0.3, random_state= 0 , shuffle=True , stratify=y)    # 0
es = EarlyStopping(monitor='val_loss', mode='min' , patience= 10 , restore_best_weights=True , verbose= 1 )


# print(y_train.shape)            # (67405, 7) // print(y_train.shape) = output 값 구하는 법


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)



#2
# model = Sequential()
# model.add(Dense(1024 ,input_dim= 13))
# model.add(Dense(512))
# model.add(Dense(256,activation= 'relu'))
# model.add(Dense(128, activation= 'relu'))
# model.add(Dense(64,activation= 'relu'))
# model.add(Dense(32,activation= 'relu'))
# model.add(Dense(7,activation='softmax'))


# #3
# from keras.callbacks import EarlyStopping ,ModelCheckpoint
# mcp = ModelCheckpoint(monitor='val_loss', mode='min' , verbose=1, save_best_only=True , filepath=  'c:/_data/_save/MCP/keras26_MCP11.hdf5'   )


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train,y_train, epochs = 10000000 , batch_size= 10000 , validation_split=0.2 , callbacks = [es,mcp] , verbose= 2 )

model = load_model('c:/_data/_save/MCP/keras26_MCP11.hdf5')

#4
loss = model.evaluate(x_test,y_test)

y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
arg_pre = np.argmax(y_predict,axis=1)
arg_test = np.argmax(y_test,axis=1)

submit =  np.argmax(y_submit,axis=1)

y_submit = encoder.inverse_transform(submit)       # inverse_transform 처리하거나, 뽑을라면 argmax처리를 해줘야한다.
submission_csv['대출등급'] = y_submit
submission_csv.to_csv(path+'submission_0115_3.csv', index = False)



def f1(arg_test,arg_pre) :
    return f1_score(arg_test,arg_pre, average='macro')
f1 = f1(arg_test,arg_pre)

def acc(arg_test,arg_pre) :
    return accuracy_score(arg_test,arg_pre)
acc = acc(arg_test,arg_pre)



submission_csv.to_csv(path+'submission_0116.csv', index = False)


print('y_submit = ', y_submit)

print('loss = ',loss)
print("f1 = ",f1)


# y_submit =  ['B' 'B' 'B' ... 'B' 'C' 'B']
# loss =  [222.89456176757812, 0.43975216150283813]
# f1 =  0.30516756362369907

# Epoch 357: early stopping
# 753/753 [==============================] - 1s 732us/step - loss: 48.2262 - acc: 0.4793
# 2007/2007 [==============================] - 1s 695us/step
# 753/753 [==============================] - 1s 683us/step
# y_submit =  ['C' 'A' 'A' ... 'C' 'C' 'A']
# loss =  [48.226158142089844, 0.47931379079818726]
# f1 =  0.37544917638611197

# 2007/2007 [==============================] - 2s 812us/step
# 753/753 [==============================] - 1s 840us/step
# y_submit =  ['B' 'A' 'A' ... 'D' 'B' 'A']
# loss =  [11.950640678405762, 0.4693860709667206]
# f1 =  0.4320287071351898


# MinMaxScaler
# Epoch 2274: early stopping
# 903/903 [==============================] - 1s 576us/step - loss: 0.4005 - acc: 0.8647
# 2007/2007 [==============================] - 1s 553us/step
# 903/903 [==============================] - 1s 571us/step
# y_submit =  ['B' 'B' 'A' ... 'D' 'C' 'A']
# loss =  [0.400475412607193, 0.8646543622016907]
# f1 =  0.8219496330591821

# StandardScaler
# Epoch 2172: early stopping
# 903/903 [==============================] - 1s 593us/step - loss: 0.3664 - acc: 0.8739
# 2007/2007 [==============================] - 1s 540us/step
# 903/903 [==============================] - 1s 560us/step
# y_submit =  ['B' 'A' 'A' ... 'D' 'C' 'A']
# loss =  [0.3664281964302063, 0.8739312291145325]
# f1 =  0.8449073685924191

# MaxAbsScaler
# Epoch 2230: early stopping
# 903/903 [==============================] - 1s 592us/step - loss: 0.4184 - acc: 0.8541
# 2007/2007 [==============================] - 1s 557us/step
# 903/903 [==============================] - 1s 560us/step
# y_submit =  ['B' 'B' 'A' ... 'D' 'C' 'A']
# loss =  [0.41843244433403015, 0.8540620803833008]
# f1 =  0.7846639869016788

# RobustScaler
# Epoch 2237: early stopping
# 903/903 [==============================] - 1s 571us/step - loss: 0.3488 - acc: 0.8790
# 2007/2007 [==============================] - 1s 548us/step
# 903/903 [==============================] - 0s 537us/step
# y_submit =  ['B' 'A' 'A' ... 'D' 'C' 'A']
# loss =  [0.34875673055648804, 0.8790196776390076]
# f1 =  0.8497940978155077




