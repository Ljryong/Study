from keras.models import Sequential
from keras.layers import Dense , Dropout , LeakyReLU , BatchNormalization
from keras.callbacks import EarlyStopping , ModelCheckpoint ,ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder , LabelEncoder , MinMaxScaler
import datetime
from imblearn.over_sampling import SMOTE
import time

#1
path = 'c:/_data/dacon/dechul//'
train_csv = pd.read_csv(path+ 'train.csv' , index_col= 0)
test_csv = pd.read_csv(path+ 'test.csv' , index_col= 0)
submission_csv = pd.read_csv(path+ 'sample_submission.csv')

encoder = LabelEncoder()
encoder.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = encoder.transform(train_csv['주택소유상태'])

encoder.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])

encoder.fit(train_csv['대출등급'])
train_csv['대출등급'] = encoder.transform(train_csv['대출등급'])

date = datetime.datetime.now()
date = date.strftime('%m%d-%H%M')
path = 'c:/_data/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path , 'k28_11_', date , '_', filename ])

train_csv['주택소유상태'] = train_csv['주택소유상태'].replace({'MORTGAGE' : 0 , 'OWN' : 1 , 'RENT': 2 , 'ANY' : 0}).astype(float)
test_csv['주택소유상태'] = test_csv['주택소유상태'].replace({'MORTGAGE' : 0 , 'OWN' : 1 , 'RENT': 2}).astype(float)

train_csv['대출목적'] = train_csv['대출목적'].replace({'부채 통합' : 0 , '주택 개선' : 2 , '주요 구매': 4 , '휴가' : 9  
                                                     , '의료' : 5 , '자동차' : 6 , '신용 카드' : 1 , '기타' : 3 , '주택개선' : 8,
                                                      '소규모 사업' : 7 , '이사' :  12 , '주택': 10 , '재생 에너지' : 11 })
test_csv['대출목적'] = test_csv['대출목적'].replace({'부채 통합' : 0 , '주택 개선' : 2 , '주요 구매': 4 , '휴가' : 9 ,
                                             '의료' : 5 , '자동차' : 6 , '신용 카드' : 1 , '기타' : 3 , '주택개선' : 8,
                                             '소규모 사업' : 7 , '이사' :  12 , '주택': 10 , '재생 에너지' : 11 , 
                                             '결혼' : 2 })

# 결혼은 train에 없는 라벨이다. 그래서 12 로 두든 2로 두든 아니면 없애든 값이 좋은걸로 비교해보면 된다.
train_csv['대출기간'] = train_csv['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)
test_csv['대출기간'] = test_csv['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)

train_csv['근로기간'] = train_csv['근로기간'].replace({'10+ years' : 11 ,            # 10으로 둘지 그 이상으로 둘지 고민
                                                      '2 years' : 2 , '< 1 year' : 0.7 , '3 years' : 3 , '1 year' : 1 ,
                                                      'Unknown' : 0 , '5 years' : 5 , '4 years' : 4 , '8 years' : 8 ,
                                                      '6 years' : 6 , '7 years' : 7 , '9 years' : 9 , '10+years' : 11,
                                                      '<1 year' : 0.7 , '3' : 3 , '1 years' : 1 })

test_csv['근로기간'] = test_csv['근로기간'].replace({'10+ years' : 11 ,            # 10으로 둘지 그 이상으로 둘지 고민
                                                      '2 years' : 2 , '< 1 year' : 0.7 , '3 years' : 3 , '1 year' : 1 ,
                                                      'Unknown' : 0 , '5 years' : 5 , '4 years' : 4 , '8 years' : 8 ,
                                                      '6 years' : 6 , '7 years' : 7 , '9 years' : 9 , '10+years' : 11,
                                                      '<1 year' : 0.7 , '3' : 3 , '1 years' : 1 })

# train_csv = train_csv[train_csv['주택소유상태'] != 'ANY' ] 
# test_csv = test_csv[test_csv['대출목적'] != '결혼' ] 

# print(train_csv['대출기간'])

print(pd.value_counts(test_csv['대출목적']))       # pd.value_counts() = 컬럼의 이름과 수를 알 수 있다.

x = train_csv.drop(['대출등급'],axis = 1 )
y = train_csv['대출등급']



y = y.values.reshape(-1)       # (96294, 1)


# # print(y.shape) 
# ohe = OneHotEncoder(sparse = False)
# ohe.fit(y)
# y_ohe = ohe.transform(y) 



x_train ,x_test , y_train , y_test = train_test_split(x,y,test_size = 0.25, random_state= 730501 , shuffle=True , stratify=y)    # 0 1502
es = EarlyStopping(monitor='val_loss', mode='min' , patience= 50 , restore_best_weights=True , verbose= 0 )




# print(y_train.shape)            # (67405, 7) // print(y_train.shape) = output 값 구하는 법


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
test_csv = scaler.transform(test_csv)        # test_csv도 같이 학습 시켜줘야 값이 나온다. 안해주면 소용이 없다.



#2
'''

model = Sequential()
model.add(Dense(102 ,input_shape= (13,),activation='swish'))
model.add(Dense(15,activation= 'swish'))
model.add(Dense(132,activation= 'swish'))
model.add(Dense(13, activation= 'swish'))
model.add(Dense(64,activation= 'swish'))
model.add(Dense(10,activation= 'swish'))
model.add(Dense(80, activation= 'swish'))
model.add(Dense(9,activation= 'swish'))
model.add(Dense(67,activation= 'swish'))
model.add(Dense(13,activation= 'swish'))
model.add(Dense(37,activation= 'swish'))
model.add(Dense(78,activation= 'swish'))
model.add(Dense(8,activation= 'swish'))
model.add(Dense(78,activation= 'swish'))
model.add(Dense(13, activation= 'swish'))
model.add(Dense(64,activation= 'swish'))
model.add(Dense(7,activation='softmax'))

'''






#3
from keras.callbacks import EarlyStopping ,ModelCheckpoint
# mcp = ModelCheckpoint(monitor='val_loss', mode='min' , verbose=1, save_best_only=True , filepath=  filepath   )
from keras.optimizers import Adam
learning_rates = [ 1.0, 0.1, 0.01, 0.001, 0.0001 ]

for learning_rate in learning_rates :
    model = Sequential()
    model.add(Dense(64 ,input_shape= (13,),activation='swish'))
    model.add(Dense(16,activation= 'swish'))
    model.add(Dense(64,activation= 'swish'))
    model.add(Dense(16, activation= 'swish'))
    model.add(Dense(64,activation= 'swish'))
    model.add(Dense(16,activation= 'swish'))
    model.add(Dense(32,activation= 'swish'))
    model.add(Dense(16, activation= 'swish'))
    model.add(Dense(32,activation= 'swish'))
    model.add(Dense(16,activation= 'swish'))
    model.add(Dense(32,activation= 'swish'))
    model.add(Dense(16, activation= 'swish'))
    model.add(Dense(64,activation= 'swish'))
    model.add(Dense(32,activation= 'swish'))
    model.add(Dense(16,activation= 'swish'))
    model.add(Dense(32,activation= 'swish'))
    model.add(Dense(16,activation= 'swish'))
    model.add(Dense(32,activation= 'swish'))
    model.add(Dense(16, activation= 'swish'))
    model.add(Dense(32,activation= 'swish'))
    model.add(Dense(16,activation= 'swish'))
    model.add(Dense(32,activation= 'swish'))
    model.add(Dense(16, activation= 'swish'))
    model.add(Dense(64,activation= 'swish'))
    model.add(Dense(32,activation= 'swish'))
    model.add(Dense(16,activation= 'swish'))
    model.add(Dense(32,activation= 'swish'))
    model.add(Dense(7,activation='softmax'))
    
    rlr = ReduceLROnPlateau(monitor='val_loss' , mode='auto' , patience= 20 , verbose= 1 , 
                            factor=0.5          # 갱신이 없으면 learning rate를 내가 지정한 수치(0.5) 만큼 나눈다
                                                # learning rate 의 default = 0.001
                                                # 이걸 쓰려면 default 보다 높게 잡고 많이 내려간 뒤 낮아지는게 좋음
                            )

    hist = model.fit(x_train,y_train, epochs = 1000 , batch_size= 32 , validation_split=0.2 , callbacks = [es, rlr] , verbose = 1 )
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate) , metrics=['acc'])
    model.fit(x_train,y_train, epochs = 10000000 , batch_size= 700 , validation_split=0.15 , callbacks = [es] , verbose= 0 )

    #4
    loss = model.evaluate(x_test,y_test)
    y_predict = model.predict(x_test)
    arg_pre = np.argmax(y_predict,axis=1)
    # arg_test = np.argmax(y_test,axis=1)

    y_predict = model.predict(x_test,verbose=0)
    print('lr : {0}, 로스 : {1} '.format(learning_rate,loss))
    acc = accuracy_score(y_test,arg_pre)
    print('lr : {0} , ACC : {1} '.format(learning_rate,acc))



'''
lr : 1.0, 로스 : [nan, 0.1741712987422943] 
lr : 1.0 , ACC : 0.17417130514247736

lr : 0.1, 로스 : [nan, 0.1741712987422943] 
lr : 0.1 , ACC : 0.17417130514247736

lr : 0.01, 로스 : [nan, 0.1741712987422943] 
lr : 0.01 , ACC : 0.17417130514247736

lr : 0.001, 로스 : [nan, 0.1741712987422943] 
lr : 0.001 , ACC : 0.17417130514247736

lr : 0.0001, 로스 : [nan, 0.1741712987422943] 
lr : 0.0001 , ACC : 0.17417130514247736
'''