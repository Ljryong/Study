from keras.preprocessing.text import Tokenizer
import os
import random as rn
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, MinMaxScaler, Normalizer, RobustScaler, StandardScaler, MaxAbsScaler
from sklearn.experimental import enable_halving_search_cv #정식버전이 아님!
from sklearn.model_selection import train_test_split,GridSearchCV,HalvingGridSearchCV,RandomizedSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
le = LabelEncoder()
import datetime
import time
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline,Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

SEED1 = 46
SEED2 = 45
SEED3 = 46

tf.random.set_seed(SEED1)  
np.random.seed(SEED2)
rn.seed(SEED3)

start_time = time.time()


#1. 데이터

path = 'c:/_data/kaggle/fat//'

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

#print(train_csv)
#print(test_csv)
#print(submission_csv)

# print(train_csv.shape) #(20758, 17)
# print(test_csv.shape) #(13840, 16)
# print(submission_csv.shape) #(13840, 2)

#print(train_csv.columns) #Index(['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
#       'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
#       'CALC', 'MTRANS', 'NObeyesdad'],



#print(train_csv['NObeyesdad'].value_counts())
# Obesity_Type_III       4046
# Obesity_Type_II        3248
# Normal_Weight          3082
# Obesity_Type_I         2910
# Insufficient_Weight    2523
# Overweight_Level_II    2522
# Overweight_Level_I     2427


##############데이터 전처리###############
le = LabelEncoder()



train_csv['BMI'] = train_csv['Weight'] / (train_csv['Height'] ** 2)
#train_csv['WIR'] = train_csv['Weight'] / train_csv['CH2O']
train_csv['bmioncp'] = train_csv['BMI'] / train_csv['NCP']


test_csv['BMI'] = test_csv['Weight'] / (test_csv['Height'] ** 2)
#test_csv['WIR'] = test_csv['Weight'] / test_csv['CH2O']
test_csv['bmioncp'] = test_csv['BMI'] / test_csv['NCP']

#Gender
train_csv['Gender']= train_csv['Gender'].str.replace("Male","0")
train_csv['Gender']= train_csv['Gender'].str.replace("Female","1")
test_csv['Gender']= test_csv['Gender'].str.replace("Male","0")
test_csv['Gender']= test_csv['Gender'].str.replace("Female","1")

# print(train_csv['Gender'])
# print(test_csv['Gender'])



#family_history_with_overweight
train_csv['family_history_with_overweight']= train_csv['family_history_with_overweight'].str.replace("yes","0")
train_csv['family_history_with_overweight']= train_csv['family_history_with_overweight'].str.replace("no","1")
test_csv['family_history_with_overweight']= test_csv['family_history_with_overweight'].str.replace("yes","0")
test_csv['family_history_with_overweight']= test_csv['family_history_with_overweight'].str.replace("no","1")

# print(train_csv['family_history_with_overweight'])
# print(test_csv['family_history_with_overweight'])

train_csv['FAVC']= train_csv['FAVC'].str.replace("yes","0")
train_csv['FAVC']= train_csv['FAVC'].str.replace("no","1")
test_csv['FAVC']= test_csv['FAVC'].str.replace("yes","0")
test_csv['FAVC']= test_csv['FAVC'].str.replace("no","1")

#print(train_csv['FAVC'])
#print(test_csv['FAVC'])
#print(np.unique(train_csv['FAVC'], return_counts= True))
#print(np.unique(test_csv['FAVC'], return_counts= True))


#print(np.unique(train_csv['CAEC'], return_counts= True))
train_csv['CAEC']= train_csv['CAEC'].str.replace("Always","0")
train_csv['CAEC']= train_csv['CAEC'].str.replace("Frequently","1")
train_csv['CAEC']= train_csv['CAEC'].str.replace("Sometimes","2")
train_csv['CAEC']= train_csv['CAEC'].str.replace("no","3")

test_csv['CAEC']= test_csv['CAEC'].str.replace("Always","0")
test_csv['CAEC']= test_csv['CAEC'].str.replace("Frequently","1")
test_csv['CAEC']= test_csv['CAEC'].str.replace("Sometimes","2")
test_csv['CAEC']= test_csv['CAEC'].str.replace("no","3")
#print(np.unique(train_csv['CAEC'], return_counts= True))
#print(np.unique(test_csv['CAEC'], return_counts= True))


#print(np.unique(test_csv['SMOKE'], return_counts= True))
train_csv['SMOKE']= train_csv['SMOKE'].str.replace("yes","0")
train_csv['SMOKE']= train_csv['SMOKE'].str.replace("no","1")
test_csv['SMOKE']= test_csv['SMOKE'].str.replace("yes","0")
test_csv['SMOKE']= test_csv['SMOKE'].str.replace("no","1")

#print(np.unique(train_csv['SMOKE'], return_counts= True))
#print(np.unique(test_csv['SMOKE'], return_counts= True))

#print(np.unique(train_csv['SCC'], return_counts= True))
train_csv['SCC']= train_csv['SCC'].str.replace("yes","0")
train_csv['SCC']= train_csv['SCC'].str.replace("no","1")
test_csv['SCC']= test_csv['SCC'].str.replace("yes","0")
test_csv['SCC']= test_csv['SCC'].str.replace("no","1")
#print(np.unique(test_csv['SCC'], return_counts= True))


#print(np.unique(test_csv['CALC'], return_counts= True))
test_csv['CALC']= test_csv['CALC'].str.replace("Always","1")
test_csv['CALC']= test_csv['CALC'].str.replace("Frequently","1")
test_csv['CALC']= test_csv['CALC'].str.replace("Sometimes","2")
test_csv['CALC']= test_csv['CALC'].str.replace("no","3")

#print(np.unique(train_csv['CALC'], return_counts= True))
train_csv['CALC']= train_csv['CALC'].str.replace("Always","0")
train_csv['CALC']= train_csv['CALC'].str.replace("Frequently","1")
train_csv['CALC']= train_csv['CALC'].str.replace("Sometimes","2")
train_csv['CALC']= train_csv['CALC'].str.replace("no","3")
#print(np.unique(train_csv['CALC'], return_counts= True))


#print(np.unique(train_csv['MTRANS'], return_counts= True))
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Automobile","0")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Bike","1")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Motorbike","2")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Public_Transportation","3")
train_csv['MTRANS']= train_csv['MTRANS'].str.replace("Walking","4")

test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Automobile","0")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Bike","1")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Motorbike","2")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Public_Transportation","3")
test_csv['MTRANS']= test_csv['MTRANS'].str.replace("Walking","4")



#print(np.unique(train_csv['MTRANS'], return_counts= True))
#print(np.unique(test_csv['MTRANS'], return_counts= True))


#print(test_csv.isnull().sum()) #없음.
#print(train_csv.isnull().sum()) #없음.

# (['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
#        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
#        'CALC', 'MTRANS', 'NObeyesdad', 'BMI'],


x = train_csv.drop(['NObeyesdad'], axis = 1)

def fit_outlier(x):  
    x = pd.DataFrame(x)
    for label in x:
        series = x[label]
        q1 = series.quantile(0.25)      
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + iqr
        lower_bound = q1 - iqr
        
        series[series > upper_bound] = np.nan
        series[series < lower_bound] = np.nan
        print(series.isna().sum())
        series = series.interpolate()
        x[label] = series
        
    x = x.fillna(x.median())
    return x

y = train_csv['NObeyesdad']
y = le.fit_transform(y)
#test_csv = test_csv.drop(['FAVC'], axis = 1)



from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
import time


mms = MinMaxScaler()
# mms = MinMaxScaler(feature_range=(0,1))
# mms = StandardScaler()
# mms = MaxAbsScaler()
# mms = RobustScaler()

mms.fit(x)
x = mms.transform(x)
test_csv=mms.transform(test_csv)










x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, shuffle=True, random_state=SEED1, stratify= y)
#5 9158 19 9145

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"n_estimators": [1400, 2000], "max_depth": [14, 10, 12], "min_samples_leaf": [3,1],"learning_rate" : [0.1, 0.01, 0.005], "gamma": [0,0.15]},
    {"n_estimators": [1500, 1900], "max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 1],"learning_rate" : [0.1, 0.01, 0.005],"gamma": [0,0.15]},
    {"n_estimators": [1600, 1800], "min_samples_leaf": [3, 5, 7], "min_samples_split": [2, 3, 4],"learning_rate" : [0.1, 0.01, 0.005],"gamma": [0,0.15]},
    {"n_estimators": [2200, 1700], "min_samples_split": [2, 3, 5], "subsample": [0, 0.3],"learning_rate" : [0.1, 0.01, 0.005],"gamma": [0,0.21]},
    {"n_estimators": [2300, 2100], "reg_alpha": [0.2351, 0.135], "min_samples_leaf": [2, 3, 5], "learning_rate" : [0.1, 0.01, 0.005], "max_depth" : [4, 8, 10, 11]}]





model = RandomizedSearchCV(LGBMClassifier(), 
                     parameters, 
                     cv=kfold, 
                     verbose=1, 
                     refit= True, #디폴트 트루~
                     random_state= 123,
                     )#n_jobs=-1) #CPU 다 쓴다!




#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


#4. 평가, 예측
from sklearn.metrics import accuracy_score
best_predict = model.best_estimator_.predict(x_test)
best_acc_score = accuracy_score(y_test, best_predict)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score :', model.best_score_)                #train의 결과이기 때문에 절대적으로 믿을 수 는 없음.

results = model.score(x_test, y_test)

print('score :', results)

y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적튠 ACC :", accuracy_score(y_test, y_predict))

print("걸린시간 :", round(end_time - start_time, 2), "초")


y_submit = model.predict(test_csv)
y_submit = le.inverse_transform(y_submit)



end_time = time.time()

print('걸린시간 :', round(end_time - start_time,2),'초')
import datetime
dt = datetime.datetime.now()
submission_csv['NObeyesdad'] = y_submit

submission_csv.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_.csv",index=False)


#이전 성적

# score : 0.9205202312138728
# acc : 0.9205202312138728
# 0.9205202312138728

#이상치.결측치 처리 후 성적

# score : 0.9210019267822736
# acc : 0.9210019267822736
# 0.9210019267822736