from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

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
encoder.fit(train_csv['대출목적'])
train_csv['대출목적'] = encoder.transform(train_csv['대출목적'])
encoder.fit(train_csv['대출기간'])
train_csv['대출기간'] = encoder.transform(train_csv['대출기간'])
encoder.fit(train_csv['근로기간'])
train_csv['근로기간'] = encoder.transform(train_csv['근로기간'])



encoder.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])
encoder.fit(test_csv['대출목적'])
test_csv['대출목적'] = encoder.transform(test_csv['대출목적'])
encoder.fit(test_csv['대출기간'])
test_csv['대출기간'] = encoder.transform(test_csv['대출기간'])
encoder.fit(test_csv['근로기간'])
test_csv['근로기간'] = encoder.transform(test_csv['근로기간'])

encoder.fit(train_csv['대출등급'])
train_csv['대출등급'] = encoder.transform(train_csv['대출등급'])




print(train_csv.dtypes)
print(test_csv.dtypes)

# encoder.fit(test_csv['대출목적'])
# encoder.fit(train_csv['대출목적'])
# encoder.fit(train_csv['대출기간'])
# encoder.fit(test_csv['대출기간'])
# encoder.fit(train_csv['근로기간'])
# encoder.fit(test_csv['근로기간'])





# train_csv['주택소유상태'] = train_csv['주택소유상태'].replace({'MORTGAGE' : 0 , 'OWN' : 1 , 'RENT': 2 , 'ANY' : 3}).astype(float)
# test_csv['주택소유상태'] = test_csv['주택소유상태'].replace({'MORTGAGE' : 0 , 'OWN' : 1 , 'RENT': 2}).astype(float)

# train_csv['대출목적'] = train_csv['대출목적'].replace({'부채 통합' : 0 , '주택 개선' : 2 , '주요 구매': 4 , '휴가' : 9  
#                                                      , '의료' : 5 , '자동차' : 6 , '신용 카드' : 1 , '기타' : 3 , '주택개선' : 8,
#                                                       '소규모 사업' : 7 , '이사' :  8 , '주택': 10 , '재생 에너지' : 11 })
# test_csv['대출목적'] = test_csv['대출목적'].replace({'부채 통합' : 0 , '주택 개선' : 2 , '주요 구매': 4 , '휴가' : 9 ,
#                                              '의료' : 5 , '자동차' : 6 , '신용 카드' : 1 , '기타' : 3 , '주택개선' : 8,
#                                              '소규모 사업' : 7 , '이사' :  8 , '주택': 10 , '재생 에너지' : 11 , 
#                                              '결혼' : 0 })
# # 결혼은 train에 없는 라벨이다. 그래서 12 로 두든 2로 두든 아니면 없애든 값이 좋은걸로 비교해보면 된다.
# train_csv['대출기간'] = train_csv['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)
# test_csv['대출기간'] = test_csv['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)

# train_csv['근로기간'] = train_csv['근로기간'].replace({'10+ years' : 10 ,            # 10으로 둘지 그 이상으로 둘지 고민
#                                                       '2 years' : 2 , '< 1 year' : 0.7 , '3 years' : 3.5 , '1 year' : 1 ,
#                                                       'Unknown' : 0 , '5 years' : 5 , '4 years' : 4 , '8 years' : 8 ,
#                                                       '6 years' : 6 , '7 years' : 7 , '9 years' : 9 , '10+years' : 11,
#                                                       '<1 year' : 0.5 , '3' : 3 , '1 years' : 1.5 })

# test_csv['근로기간'] = test_csv['근로기간'].replace({'10+ years' : 10 ,            # 10으로 둘지 그 이상으로 둘지 고민
#                                                       '2 years' : 2 , '< 1 year' : 0.7 , '3 years' : 3.5 , '1 year' : 1 ,
#                                                       'Unknown' : 0 , '5 years' : 5 , '4 years' : 4 , '8 years' : 8 ,
#                                                       '6 years' : 6 , '7 years' : 7 , '9 years' : 9 , '10+years' : 11,
#                                                       '<1 year' : 0.5 , '3' : 3 , '1 years' : 1.5 })


# print(train_csv['대출기간'])

# print(pd.value_counts(test_csv['근로기간']))       # pd.value_counts() = 컬럼의 이름과 수를 알 수 있다.

x = train_csv.drop(['대출등급'],axis = 1 )
y = train_csv['대출등급']
print(train_csv.dtypes)

print(y.shape)      # (96294,)

y = y.values.reshape(-1,1)       # (96294, 1)

# print(y.shape) 
ohe = OneHotEncoder(sparse = False)
ohe.fit(y)
y_ohe = ohe.transform(y) 


x_train ,x_test , y_train , y_test = train_test_split(x,y_ohe,test_size = 0.3, random_state= 56 , shuffle=True , stratify=y)
es = EarlyStopping(monitor='val_loss', mode='min' , patience= 100 , restore_best_weights=True , verbose= 1 )


print(y_train.shape)            # (67405, 7) // print(y_train.shape) = output 값 구하는 법

#2
model = Sequential()
model.add(Dense(2048,input_dim= 13))
model.add(Dense(1024))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(7,activation='softmax'))


#3
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs = 10000000 , batch_size= 3000 , validation_split=0.2 , callbacks = [es] , verbose= 1 )


#4
loss = model.evaluate(x_test,y_test)

y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
arg_pre = np.argmax(y_predict,axis=1)
arg_test = np.argmax(y_test,axis=1)

submit =  np.argmax(y_submit,axis=1)

y_submit = encoder.inverse_transform(submit)       # inverse_transform 처리하거나 뽑을라면 argmax처리를 해줘야한다.
submission_csv['대출등급'] = y_submit
submission_csv.to_csv(path+'submission_0115.csv', index = False)





def f1(arg_test,arg_pre) :
    return f1_score(arg_test,arg_pre, average='macro')
f1 = f1(arg_test,arg_pre)

def acc(arg_test,arg_pre) :
    return accuracy_score(arg_test,arg_pre)
acc = acc(arg_test,arg_pre)



submission_csv.to_csv(path+'submission_0115.csv', index = False)


print('y_submit = ', y_submit)

print('loss = ',loss)

print("f1 = ",f1)


# y_submit =  ['B' 'B' 'B' ... 'B' 'C' 'B']
# loss =  [222.89456176757812, 0.43975216150283813]
# f1 =  0.30516756362369907

