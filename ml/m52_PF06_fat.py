import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split , StratifiedKFold , RandomizedSearchCV , GridSearchCV
import numpy as np
import dask
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import warnings
warnings.filterwarnings('ignore')
import time
from sklearn.preprocessing import PolynomialFeatures

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


train_csv = train_csv.drop(['SMOKE'],axis=1)
test_csv = test_csv.drop(['SMOKE'],axis=1)

x = train_csv.drop(['NObeyesdad'], axis= 1)
y = train_csv['NObeyesdad']

from sklearn.preprocessing import MinMaxScaler , StandardScaler , MaxAbsScaler , RobustScaler

df = pd.DataFrame(x , columns = x.columns)
print(df)
df['target(Y)'] = y
print(df)

print('=================== 상관계수 히트맵 =====================')
print(df.corr())

pf = PolynomialFeatures(degree=2 , include_bias=False )
x1 = pf.fit_transform(x)

x_train , x_test , y_train , y_test = train_test_split(x1,y, random_state= 980909 , test_size=0.3 , shuffle=True , stratify=y )

# scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

x_train = np.array(x_train)
x_test = np.array(x_test)

#2 모델
from sklearn.ensemble import BaggingClassifier , RandomForestClassifier , VotingClassifier , StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model = RandomForestClassifier()


#3 훈련
model.fit(x_train,y_train)

#4 평가, 예측
from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print('model.score : ' , model.score(x_test,y_test) )
print('accuracy : ' , accuracy_score(y_test,y_pred) )


# xgb
# ACC 0.9059087989723827
# True
# ACC 0.8485870263326911
# False
# ACC 0.848747591522158


# soft
# ACC 0.9047848426461144
# hard
# ACC 0.901252408477842

# model.score :  0.8994861913937059
# accuracy :  0.8994861913937059

# PolynomialFeatures
# model.score :  0.8975594091201028
# accuracy :  0.8975594091201028