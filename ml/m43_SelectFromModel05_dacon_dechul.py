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


parameters = {'n_estimators' : 1000, 
              'learning_rate' : 0.1,
              'max_depth': 3,               # 트리 깊이
              'gamma' : 0,
              'min_child_weight' : 0,       # 드랍 아웃 개념
              'subsample' : 0.4,
              'colsample_bytree' : 0.8,
              'colsample_bylevel' : 0.7,
              'colsample_bynode' : 1,
              'reg_alpha' : 0,              # 알파, 람다 , L1 , L2 규제
              'reg_lamda' : 1,
              'random_state' : 3377,
              'verbose' : 0,
              }

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
from sklearn.feature_selection import SelectFromModel
warnings.filterwarnings('ignore')
thresholds = np.sort(model.feature_importances_)
print(thresholds)

for i in thresholds:                                                    # 제일 작은것들을 먼저 없애줌
    # i 보다 크거나 같은 것만 남음 
    selection =  SelectFromModel(model, threshold=i ,prefit=False)        # selectionws은 인스턴스(변수)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(i ,'\t변형된 x_train',select_x_train.shape, i ,'변형된 x_test',select_x_test.shape)
    
    select_model = XGBClassifier()
    select_model.set_params(early_stopping_rounds = 10 , **parameters ,
                            # eval_metric = 'logloss'
                            )
    
    select_model.fit(select_x_train,y_train  , eval_set = [(select_x_train , y_train  ),(select_x_test,y_test  )], verbose = 0 ) 
    
    
    select_y_predict = select_model.predict(select_x_test)
    score = accuracy_score(y_test , select_y_predict)
    
    print("Thredsholds=%.3f, n=%d, ACC: %.2f%%" %(i, select_x_train.shape[1], score*100))



# Thredsholds=0.011, n=13, ACC: 81.92%
# Thredsholds=0.011, n=12, ACC: 81.50%
# Thredsholds=0.012, n=11, ACC: 81.28%
# Thredsholds=0.015, n=10, ACC: 81.71%
# Thredsholds=0.016, n=9, ACC: 81.60%
# Thredsholds=0.018, n=8, ACC: 81.97%
# Thredsholds=0.020, n=7, ACC: 81.61%
# Thredsholds=0.030, n=6, ACC: 80.20%
# Thredsholds=0.034, n=5, ACC: 81.83%
# Thredsholds=0.045, n=4, ACC: 82.96%
# Thredsholds=0.186, n=3, ACC: 80.11%
# Thredsholds=0.192, n=2, ACC: 37.04%
# Thredsholds=0.410, n=1, ACC: 34.76%