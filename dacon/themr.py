import numpy as np
import random 
import os
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split , GridSearchCV , RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(980909)      # 98099

path = 'C:\_data\dacon\소득\\'

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

x = train.drop(columns=['ID', 'Income'])
y = train['Income']

test_csv = test.drop(columns=['ID'])

# 라벨인코더 
encoding_target = list(x.dtypes[x.dtypes == "object"].index)

for i in encoding_target:
    le = LabelEncoder()
    
    # train과 test 데이터셋에서 해당 열의 모든 값을 문자열로 변환
    x[i] = x[i].astype(str)
    test_csv[i] = test_csv[i].astype(str)
    
    le.fit(x[i])
    x[i] = le.transform(x[i])
    
    # test 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
    for case in np.unique(test_csv[i]):
        if case not in le.classes_: 
            le.classes_ = np.append(le.classes_, case)
    
    test_csv[i] = le.transform(test_csv[i])


# 데이터를 섞음
# x_shuffled, y_shuffled = shuffle(x, y, random_state=220118)

# scaler = MinMaxScaler()
scaler = StandardScaler()

scaler.fit_transform(x)

grid_params = {
    # 'iterations': 1000,
    'learning_rate': [1.0,0.1,0.5,0.01],
    'early_stopping_rounds': [30],  
    'max_depth' : [10,6] ,
    # 'reduce_learning_rate' : True,
    'random_state' : [980909]
}

params = {
    'iterations': 3000,
    # 'learning_rate': [1.0,0.1,0.5,0.01],
    'learning_rate': 1.0,
    'early_stopping_rounds': 30,  
    'max_depth' : 10 ,
    # 'reduce_learning_rate' : True,
    # 'random_state' : 980909,
    
}


# model = XGBRegressor() 
# model = CatBoostRegressor(task_type='GPU' , loss_function='RMSE', 
#                           **params,
#                           random_state=980909
#                           ) 

model = RandomForestRegressor(**params,
                          random_state=980909)

# model = GridSearchCV(model,grid_params,cv= 3 , n_jobs= 16 ,
#                     #  n_iter = 10, 
#                      refit=True)

# model = RandomizedSearchCV(model,grid_params,cv= 3 , n_jobs= 16 ,
#                     #  n_iter = 10, 
#                      refit=True)

""" model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold ,
                    verbose=1,
                    refit=True,
                    n_jobs= -1,
                    random_state=66, 
                    n_iter= 10       ) """ 

model.fit(x, y) 

preds = model.predict(test_csv)

submission = pd.read_csv(path+'sample_submission.csv')
submission['Income'] = preds

submission.to_csv(path + 'submit.csv', index=False)

import pickle
pickle.dump(model, open(path + 'themr_save.dat' , 'wb' ))
