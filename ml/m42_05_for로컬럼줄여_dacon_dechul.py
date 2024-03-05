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

# print(pd.value_counts(test_csv['대출목적']))       # pd.value_counts() = 컬럼의 이름과 수를 알 수 있다.

x = train_csv.drop(['대출등급'],axis = 1 )
y = train_csv['대출등급']


x_train ,x_test , y_train , y_test = train_test_split(x,y,test_size = 0.25, random_state= 730501 , shuffle=True , stratify=y)    # 0 1502
es = EarlyStopping(monitor='val_loss', mode='min' , patience= 3000 , restore_best_weights=True , verbose= 1 )

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

# test_csv = np.delete(test_csv,1,axis=1)
test_csv = scaler.transform(test_csv)        # test_csv도 같이 학습 시켜줘야 값이 나온다. 안해주면 소용이 없다.


#2
from xgboost import XGBClassifier

model = XGBClassifier()


#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
print(x.shape)

# 초기 특성 중요도
import warnings
warnings.filterwarnings('ignore')
feature_importances = model.feature_importances_
sort= np.argsort(feature_importances)               # argsort 열의 번호로 반환해줌
print(sort)

removed_features = 0

# 각 반복에서 피처를 추가로 제거하면서 성능 평가
for i in range(len(model.feature_importances_) - 1):
    remove = sort[:i+1]  # 추가로 제거할 피처의 인덱스
    
    print(f"Removing features at indices: {remove}")
    
    # 해당 특성 제거
    x_train_removed = np.delete(x_train, remove, axis=1)
    x_test_removed = np.delete(x_test, remove, axis=1)

    # 모델 재구성 및 훈련
    model.fit(x_train_removed, y_train, eval_set=[(x_train_removed, y_train), (x_test_removed, y_test)],
              verbose=0, eval_metric='mlogloss', early_stopping_rounds=10)
    
    # 모델 평가
    acc = model.score(x_test_removed, y_test)
    print('Accuracy after removing features:', acc)
    
    # 제거된 피처의 개수를 누적
    removed_features += 1
    print(f"Total number of removed features: {removed_features}\n")




# n_components =  1 result 0.24374844230289938
# ==================================================
# n_components =  2 result 0.2787239345351832
# ==================================================
# n_components =  3 result 0.3708565257123868
# ==================================================
# n_components =  4 result 0.39519813907119716
# ==================================================
# n_components =  5 result 0.43108748026917004
# ==================================================
# n_components =  6 result 0.4408490487663039
# ==================================================
# n_components =  7 result 0.4485336877959625
# ==================================================
# n_components =  8 result 0.4463321425604386
# ==================================================
# n_components =  9 result 0.4501121541912437
# ==================================================
# n_components =  10 result 0.45193985212262194
# ==================================================
# n_components =  11 result 0.44097366453435244
# ==================================================
# n_components =  12 result 0.4547644761983883
# ==================================================
# n_components =  13 result 0.6250311539420121
# ==================================================

# model.score 0.6286450112154192
# (96294, 13)
# [0.18666925 0.2863345  0.37429761 0.45890175 0.53661121 0.61369071
#  0.68787408 0.75785515 0.82439995 0.88729638 0.94514538 0.97374858
#  1.        ]


# model.score 0.507227714546814
# (96294, 13)
# [0.96091054 0.99128576 0.99780485 0.99943246 0.99984281 1.        ]


# model.score 0.4680983633795796

# model.score 0.8510010800033231
# (96294, 13)
# [12 11  2  6  3  5  8  7  4  0  9 10  1]
# Removing features at indices: [12]
# Accuracy after removing features: 0.8529949322920993
# Total number of removed features: 1

# Removing features at indices: [12 11]
# Accuracy after removing features: 0.853451856774944
# Total number of removed features: 2

# Removing features at indices: [12 11  2]
# Accuracy after removing features: 0.8538257040790894
# Total number of removed features: 3

# Removing features at indices: [12 11  2  6]
# Accuracy after removing features: 0.8541580127938855
# Total number of removed features: 4

# Removing features at indices: [12 11  2  6  3]
# Accuracy after removing features: 0.8539087812577885
# Total number of removed features: 5

# Removing features at indices: [12 11  2  6  3  5]
# Accuracy after removing features: 0.8565257123868073
# Total number of removed features: 6

# Removing features at indices: [12 11  2  6  3  5  8]
# Accuracy after removing features: 0.8583949489075351
# Total number of removed features: 7

# Removing features at indices: [12 11  2  6  3  5  8  7]
# Accuracy after removing features: 0.8597657223560687
# Total number of removed features: 8

# Removing features at indices: [12 11  2  6  3  5  8  7  4]
# Accuracy after removing features: 0.8668272825454848
# Total number of removed features: 9

# Removing features at indices: [12 11  2  6  3  5  8  7  4  0]
# Accuracy after removing features: 0.8528287779347014
# Total number of removed features: 10

# Removing features at indices: [12 11  2  6  3  5  8  7  4  0  9]
# Accuracy after removing features: 0.37903962781423944
# Total number of removed features: 11

# Removing features at indices: [12 11  2  6  3  5  8  7  4  0  9 10]
# Accuracy after removing features: 0.3475533770873141
# Total number of removed features: 12