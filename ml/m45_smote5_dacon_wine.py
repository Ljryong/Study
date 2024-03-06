
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC


#1 
path = "c:/_data/dacon/wine//"

train_csv = pd.read_csv(path + "train.csv" , index_col= 0)      # index_col : 컬럼을 무시한다. //  index_col= 0 는 0번째 컬럼을 무시한다. 
test_csv = pd.read_csv(path + "test.csv" , index_col= 0)
submission_csv = pd.read_csv(path + "sample_submission.csv")


####### keras에 있는 데이터 수치화 방법 ##########
train_csv['type'] = train_csv['type'].replace({'white': 0, 'red':1})
test_csv['type'] = test_csv['type'].replace({'white': 0, 'red':1})

x = train_csv.drop(['quality'], axis = 1)
y = train_csv['quality'] -3
# print(train_csv)
# print(y.shape)          # (5497,1)

print(np.unique(y))

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

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.3 , random_state= 971 , shuffle=True , stratify= y )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=13,k_neighbors = 3 )

x_train, y_train = smote.fit_resample(x_train,y_train)

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

es = EarlyStopping(monitor='val_loss' , mode = 'min', verbose=1, patience= 100 , restore_best_weights=True )



#2
from xgboost import XGBClassifier

# model = RandomForestRegressor()
# model = RandomForestClassifier()
model = XGBClassifier()

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
pre = model.predict(x_test)
print('model.score' , result)
print('acc',accuracy_score(y_test,pre))
print('f1',f1_score(y_test,pre,average='macro'))


# SMOTE 미사용
# model.score 0.6254545454545455
# acc 0.6254545454545455
# f1 0.34920883144191406

# SMOTE 사용
# model.score 0.5981818181818181
# acc 0.5981818181818181
# f1 0.35361668792533535