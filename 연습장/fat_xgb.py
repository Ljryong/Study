import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split , StratifiedKFold , RandomizedSearchCV 
import numpy as np

#1 데이터
path = 'c:/_data/kaggle/fat//'

train_csv = pd.read_csv(path + 'train.csv',index_col=0)
test_csv = pd.read_csv(path + 'test.csv',index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

le = LabelEncoder()
le.fit(train_csv['Gender'])
train_csv['Gender'] = le.transform(train_csv['Gender'])
test_csv['Gender'] = le.transform(test_csv['Gender'])

le.fit(train_csv['family_history_with_overweight'])
train_csv['family_history_with_overweight'] = le.transform(train_csv['family_history_with_overweight'])
test_csv['family_history_with_overweight'] = le.transform(test_csv['family_history_with_overweight'])

le.fit(train_csv['FAVC'])
train_csv['FAVC'] = le.transform(train_csv['FAVC'])
test_csv['FAVC'] = le.transform(test_csv['FAVC'])

le.fit(train_csv['SMOKE'])
train_csv['SMOKE'] = le.transform(train_csv['SMOKE'])
test_csv['SMOKE'] = le.transform(test_csv['SMOKE'])

le.fit(train_csv['SCC'])
train_csv['SCC'] = le.transform(train_csv['SCC'])
test_csv['SCC'] = le.transform(test_csv['SCC'])

le.fit(train_csv['NObeyesdad'])
train_csv['NObeyesdad'] = le.transform(train_csv['NObeyesdad'])

train_csv['CAEC'] = train_csv['CAEC'].replace({'Always': 0 , 'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })
test_csv['CAEC'] = test_csv['CAEC'].replace({'Always': 0 , 'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })

train_csv['CALC'] = train_csv['CALC'].replace({'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })
test_csv['CALC'] = test_csv['CALC'].replace({'Always' : 2 , 'Frequently' : 1 , 'Sometimes' : 2 , 'no' : 3 })

train_csv['MTRANS'] = train_csv['MTRANS'].replace({'Automobile': 0 , 'Bike' : 1, 'Motorbike' : 2, 'Public_Transportation' : 3,'Walking' : 4})
test_csv['MTRANS'] = test_csv['MTRANS'].replace({'Automobile': 0 , 'Bike' : 1, 'Motorbike' : 2, 'Public_Transportation' : 3,'Walking' : 4})

x = train_csv.drop(['NObeyesdad'], axis= 1)
y = train_csv['NObeyesdad']

from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , RobustScaler

x_train , x_test , y_train , y_test = train_test_split(x,y, random_state= 123456 , test_size=0.3 , shuffle=True , stratify=y )

scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

x_train = np.array(x_train)
x_test = np.array(x_test)

model = xgb.XGBClassifier()

kfold = StratifiedKFold(n_splits= 10 , shuffle=True , random_state= 1234321 )

xgb_grid = [{
    'n_estimators': np.random.randint(10,100,20),                                  # 리스트를 단일 값으로 변경
    'max_depth': np.random.randint(1,20,5),
    'learning_rate': np.random.randint(1, 5, 3),
    'min_child_weight': np.random.randint(1,50,10),                                # 리스트를 단일 값으로 변경
    'gamma': np.random.randint(0, 10, 5 ) ,
    'subsample': np.random.randint(6, 70, 10),
    'colsample_bytree': np.random.randint(6, 30, 10),
    'reg_alpha': np.random.uniform(0, 1, 10),                               # 
    'reg_lambda': np.random.uniform(0, 1, 10),
}]

# RandomizedSearchCV를 사용하여 모델을 탐색
random_search = RandomizedSearchCV(model, param_distributions=xgb_grid, n_iter= 15 , cv=kfold, random_state= 1234 )
random_search.fit(x_train, y_train)

# 최적의 하이퍼파라미터 출력
print("Best parameters found: ", random_search.best_params_)

model = xgb.XGBClassifier(*xgb_grid)

#3 훈련
model.fit(x_train,y_train)

#4 평가, 예측
# GridSearchCV 전용
from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('='*100)
acc= accuracy_score(y_test,y_predict)
print('ACC',acc)
print("Best parameters found: ", random_search.best_params_)

y_submit = model.predict(test_csv)

y_submit = le.inverse_transform(y_submit) 
submission_csv['NObeyesdad'] = y_submit

submission_csv.to_csv(path+'submission_xgb.csv', index = False)





