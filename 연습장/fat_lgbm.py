import lightgbm as lg
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

# train_csv = train_csv.drop(['SMOKE'],axis=1)
# test_csv = test_csv.drop(['SMOKE'],axis=1)

x = train_csv.drop(['NObeyesdad'], axis= 1)
y = train_csv['NObeyesdad']

from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , RobustScaler


df = pd.DataFrame(x , columns = x.columns)
print(df)
df['target(Y)'] = y
print(df)

print('=================== 상관계수 히트맵 =====================')
print(df.corr())


x_train , x_test , y_train , y_test = train_test_split(x,y, random_state= 1234 , test_size=0.3 , shuffle=True , stratify=y )

scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

kfold = StratifiedKFold(n_splits= 7 , shuffle=True , random_state= 1234 )

lgbm_grid = [{
    'n_estimators': np.random.randint(100, 300, 3),       # 랜덤으로 범위내 수를 뽑음
    'max_depth': np.random.randint(1, 50, 2),               # 랜덤으로 범위내 수를 뽑음
    'learning_rate': np.random.uniform(1e-3, 0.1, 3),      # 랜덤으로 범위내 수를 뽑음
    'min_child_samples': np.random.randint(10, 40, 3),    # 랜덤으로 범위내 수를 뽑음
    'boosting_type': ['gbdt', 'dart', 'goss', 'rf'],       # lgbm의 boosting_type은 'gbdt', 'dart', 'goss', 'rf' 중 하나
}]

# RandomizedSearchCV를 사용하여 모델을 탐색
model = RandomizedSearchCV(lg.LGBMClassifier(), lgbm_grid, n_iter= 20 , cv=kfold, random_state= 1234 )

#3 훈련
model.fit(x_train,y_train)

#4 평가, 예측
# GridSearchCV 전용
from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('='*100)
acc= accuracy_score(y_test,y_predict)
print('ACC',acc)
y_submit = model.predict(test_csv)

y_submit = le.inverse_transform(y_submit) 
submission_csv['NObeyesdad'] = y_submit

submission_csv.to_csv(path+'submission_lgbm.csv', index = False)




# lgbm
# 1234 1234 1234


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
print(sns.__version__)
print(matplotlib.__version__)      
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), 
            square=True,    
            annot=True,            
            cbar=True)             
plt.show()


