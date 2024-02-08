from keras.models import Sequential
from keras.layers import Dense , Dropout , LeakyReLU , BatchNormalization
from keras.callbacks import EarlyStopping , ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder , LabelEncoder , MinMaxScaler
import datetime
from imblearn.over_sampling import SMOTE
import time
from sklearn.svm import LinearSVC

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
es = EarlyStopping(monitor='val_loss', mode='min' , patience= 3000 , restore_best_weights=True , verbose= 1 )




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
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

allAlgorisms = all_estimators(type_filter='classifier')

from sklearn.model_selection import cross_val_predict , cross_val_score , StratifiedKFold
kfold = StratifiedKFold(n_splits=3 , random_state=123 , shuffle= True)

for name, algorithm in allAlgorisms : 
    # 에러가 떳을때 밑에 print로 넘어가고 다음이 실행됨
    try:
        #2 모델
        model = algorithm()
        #3 훈련
        scores = cross_val_score(model, x_train,y_train, cv=kfold )
        # cv = cross validation
        print('=====================',name , '=====================')
        print('Acc :',scores ,'\n 평균 acc :' , round(np.mean(scores),4) )
        # round( ,3) 소수 4번째자리에서 반올림하여 소수 3번째자리까지 나오게함
        y_predict = cross_val_predict(model,x_test,y_test,cv=kfold)
        acc = accuracy_score(y_test,y_predict)
        print('cross_val_predict_Acc',acc)
    except:
        print(name,  '은 바보 멍충이!!!')
        continue            # 에러를 무시하고 쭉 진행됨 

# y_submit = encoder.inverse_transform(submit)       # inverse_transform 처리하거나, 뽑을라면 argmax처리를 해줘야한다.
# submission_csv['대출등급'] = y_submit


# def f1(arg_test,arg_pre) :
#     return f1_score(arg_test,arg_pre, average='macro')
# f1 = f1(y_test,arg_pre)

# def acc(arg_test,arg_pre) :
#     return accuracy_score(arg_test,arg_pre)
# acc = acc(y_test,arg_pre)



# submission_csv.to_csv(path+'submission_0202.csv', index = False)


# print('y_submit = ', y_submit)

# print("f1 = ",f1)


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




# #2
# model = Sequential()
# model.add(Dense(102 ,input_shape= (13,)))
# model.add(Dense(15,activation= 'swish'))
# model.add(Dense(132,activation= 'swish'))
# model.add(Dense(13, activation= 'swish'))
# model.add(Dense(64,activation= 'swish'))
# model.add(Dense(7,activation='softmax'))
# Epoch 3597: early stopping
# 903/903 [==============================] - 1s 876us/step - loss: 0.2848 - acc: 0.9255
# 2007/2007 [==============================] - 1s 505us/step
# 903/903 [==============================] - 0s 545us/step
# y_submit =  ['B' 'B' 'A' ... 'D' 'C' 'A']
# loss =  [0.2847670018672943, 0.9255425930023193]
# f1 =  0.9094408160608632
# = 0.92



# loss =  0.4176705159092797
# 0.4176705159092797



# LinearSVC score  0.42606131095787986
# LinearSVC predict  0.42606131095787986
# Perceptron score  0.33903796627066546
# Perceptron predict  0.33903796627066546
# LogisticRegression score  0.3818227133006563
# LogisticRegression predict  0.3818227133006563
# RandomForestClassifier score  0.8079671014372352
# RandomForestClassifier predict  0.8079671014372352
# DecisionTreeClassifier score  0.8299825537924732
# DecisionTreeClassifier predict  0.8299825537924732
# KNeighborsClassifier score  0.4624491152280469
# KNeighborsClassifier predict  0.4624491152280469



# AdaBoostClassifier 의 Acc는 0.5061061726343773
# BaggingClassifier 의 Acc는 0.8659549721691452
# BernoulliNB 의 Acc는 0.37683808257871565
# CalibratedClassifierCV 의 Acc는 0.4315444047520146
# CategoricalNB 은 바보 멍청이!!!
# ClassifierChain 은 바보 멍청이!!!
# ComplementNB 은 바보 멍청이!!!
# DecisionTreeClassifier 의 Acc는 0.8305640940433663
# DummyClassifier 의 Acc는 0.299243997673839
# ExtraTreeClassifier 의 Acc는 0.42660131261942347
# ExtraTreesClassifier 의 Acc는 0.6796128603472626
# GaussianNB 의 Acc는 0.2735731494558445
# GaussianProcessClassifier 은 바보 멍청이!!!
# GradientBoostingClassifier 의 Acc는 0.7505192323668688
# HistGradientBoostingClassifier 의 Acc는 0.7897731993021517
# KNeighborsClassifier 의 Acc는 0.4624491152280469
# LabelPropagation 의 Acc는 0.452853701088311
# LabelSpreading 의 Acc는 0.4545567832516408
# LinearDiscriminantAnalysis 의 Acc는 0.4028827781008557
# LinearSVC 의 Acc는 0.4257705408324333
# LogisticRegression 의 Acc는 0.3818227133006563
# LogisticRegressionCV 의 Acc는 0.4837584115643433
# MLPClassifier 의 Acc는 0.82857024175459
# MultiOutputClassifier 은 바보 멍청이!!!
# MultinomialNB 은 바보 멍청이!!!
# NearestCentroid 의 Acc는 0.17404668937442885
# NuSVC 은 바보 멍청이!!!
# OneVsOneClassifier 은 바보 멍청이!!!
# OneVsRestClassifier 은 바보 멍청이!!!
# OutputCodeClassifier 은 바보 멍청이!!!
# PassiveAggressiveClassifier 의 Acc는 0.3514995430755172
# Perceptron 의 Acc는 0.33903796627066546
# QuadraticDiscriminantAnalysis 의 Acc는 0.30418708980643017
# RadiusNeighborsClassifier 은 바보 멍청이!!!
# RandomForestClassifier 의 Acc는 0.8075101769543906
# RidgeClassifier 의 Acc는 0.3745534601644928
# RidgeClassifierCV 의 Acc는 0.3744288443964443
# SGDClassifier 의 Acc는 0.4270166985129185
# SVC 의 Acc는 0.29961784497798455
# StackingClassifier 은 바보 멍청이!!!
# VotingClassifier 은 바보 멍청이!!!



