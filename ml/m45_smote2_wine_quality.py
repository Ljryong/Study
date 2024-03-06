import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score ,f1_score
from sklearn.ensemble import RandomForestClassifier

#1 데이터

path = 'c:/_data/dacon/wine/'

train_csv = pd.read_csv(path + 'train.csv' , index_col= 0 )
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

la = LabelEncoder()
train_csv['type'] = la.fit_transform(train_csv['type'])
test_csv['type'] = la.fit_transform(test_csv['type'])

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality']-3

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.15 , random_state= 909 , stratify=y , shuffle = True )

from imblearn.over_sampling import SMOTE        # 옛날에는 다운을 받았어야 됬는데 anaconda가 다 끌어옴
import sklearn as sk

smote = SMOTE(random_state=0, k_neighbors = 3 )
x_train , y_train = smote.fit_resample(x_train,y_train)                    # 다시 샘플링한다

print(pd.value_counts(y_train))                    # 0    53 , 1    53 , 2    53    // 53인 이유는 train과 test로 짤려서 그렇다.
print(np.unique(x_train,return_counts=True))

scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#2 모델
model = RandomForestClassifier( random_state = 42 )

#3 컴파일, 훈련
model.fit(x_train , y_train )

#4 평가
acc = model.score(x_test,y_test)
print('acc',acc)
predict = model.predict(x_test)
print('accuracy' , accuracy_score(y_test,predict) )
print('f1' , f1_score(y_test,predict , average='macro') )


# acc 0.6715151515151515
# accuracy 0.6715151515151515
# parameters = {'n_estimators' : 1000, 
#               'learning_rate' : 0.5,
#               'max_depth': 10,               # 트리 깊이
#               'gamma' : 0,
#               'min_child_weight' : 3,       # 드랍 아웃 개념
#               'subsample' : 0.4,
#               'colsample_bytree' : 0.8,
#               'colsample_bylevel' : 0.7,
#               'colsample_bynode' : 1,
#               'reg_alpha' : 0,              # 알파, 람다 , L1 , L2 규제
#             #   'reg_lamda' : 1,
#               'random_state' : 3377,
#             #   'verbose' : 0,
#               }

# SMOTE 미사용 
# acc 0.6484848484848484
# accuracy 0.6484848484848484


# SMOTE 사용
# acc 0.6484848484848484
# accuracy 0.6484848484848484
# f1 0.389225192269729