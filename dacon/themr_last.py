import numpy as np
import random
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
# from skopt import BayesSearchCV
import datetime
import matplotlib.pyplot as plt
import optuna


""" 
최고점수
random_state = 42
xgb_randomstate = 220118
cat_randomstate = 220118 """

random_state = 730501
xgb_randomstate = 730501
cat_randomstate = 730501
# lgbm_randomstate = 42

# 데이터 로드
path = 'C:\_data\dacon\소득\\'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv')

train = train.drop(['Gains', 'Losses','Dividends'], axis=1)
test = test.drop(['Gains', 'Losses', 'Dividends'], axis=1)

""" fig, axes = plt.subplots(1, 2, figsize=(10, 4))

train['Losses'].hist(bins =  50,ax=axes[0])
axes[0].set_title('Histogram')

train['Losses'].plot(kind = 'box',ax=axes[1])
axes[1].set_title('Boxplot')

plt.tight_layout()

# Show the plot
plt.show() """

# Age < 83 이상치 제거
# Gains < 10 제거
# Losses < 50 제거 
# Income < 2200 제거 
# Dividends < 1 제거 했을떄 안했을 떄 확인 필요

# Income_Status Unknown 제거 생각

# Age	Gender	Education_Status	Employment_Status	Working_Week (Yearly)	Industry_Status	Occupation_Status	
# Race	Hispanic_Origin	Martial_Status	Household_Status	Household_Summary	Citizenship	Birth_Country	Birth_Country (Father)	
# Birth_Country (Mother)	Tax_Status	Gains	Losses	Dividends	Income_Status	Income

# mean 성능 저하 / ffill  /

# train 데이터에서 Age 열의 평균값 계산
# mean_age = train['Age'].ffill()

# train.loc[train['Age'] < 83, 'Age'] = mean_age
# test.loc[test['Age'] < 83, 'Age'] = mean_age

# mean_age = train['Gains'].mean()

# train.loc[train['Gains'] < 10, 'Gains'] = mean_age
# test.loc[test['Gains'] < 10, 'Gains'] = mean_age

""" mean_age = train['Losses'].fillna(0)        # 0 으로 채울지 아니면 놔둘지 고민

train.loc[train['Losses'] < 50, 'Losses'] = mean_age
test.loc[test['Losses'] < 50, 'Losses'] = mean_age

# mean_age = train['Income'].mean()

# train.loc[train['Income'] < 2200, 'Income'] = mean_age

mean_age = train['Dividends'].mean()

train.loc[train['Dividends'] < 1, 'Dividends'] = mean_age
test.loc[test['Dividends'] < 1, 'Dividends'] = mean_age """


""" x = train.drop(columns=['Income'])
y = train['Income']



df = pd.DataFrame(x, columns=x.columns)

# df = pd.DataFrame(x , columns = train.feature_names)
print(df)
df['target(Y)'] = y
print(df)

print('=================== 상관계수 히트맵 =====================')
print(df.corr())                              

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
print(sns.__version__)
print(matplotlib.__version__)               # 3.8.0
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), 
            square=True,    
            annot=True,                       # 표안에 수치 명시
            cbar=True)                        # 사이드 바
plt.show() """


category_mapping = {
    'Householder': 'Head of Household',
    'Spouse of householder': 'Head of Household',
    'Child <18 never marr not in subfamily': 'Unmarried Child',
    'Child 18+ never marr Not in a subfamily': 'Unmarried Child',
    'Other Rel 18+ never marr not in subfamily': 'Unmarried Other Relative',
    'Other Rel 18+ ever marr not in subfamily': 'Married Other Relative',
    'Grandchild <18 never married child of subfamily Responsible Person': 'Child of Responsible Person',
    'Child <18 never married Responsible Person of subfamily': 'Child of Responsible Person'
    # 추가적인 매핑이 필요하다면 여기에 계속해서 추가할 수 있습니다.
}

# 매핑에 따라 카테고리를 새로운 그룹으로 변환
train['Household_Status'] = train['Household_Status'].map(category_mapping)
test['Household_Status'] = test['Household_Status'].map(category_mapping)

""" # Armed Forces가 포함된 행 삭제
train = train[train['Occupation_Status'] != 'Armed Forces']
test = test[test['Occupation_Status'] != 'Armed Forces']
# Industry_Status
train = train[(train['Industry_Status'] != 'Armed Forces') & (train['Industry_Status'] != 'Forestry & Fisheries')]
test = test[(test['Industry_Status'] != 'Armed Forces') & (test['Industry_Status'] != 'Forestry & Fisheries')]
"""

# 원하는 값을 사용하여 해당 행을 필터링
desired_values = ['Armed Forces']
train = train[~train['Occupation_Status'].isin(['Unknown'])]
test = test[~test['Occupation_Status'].isin(['Unknown'])]

desired_values = ['Armed Forces', 'Forestry & Fisheries']
train = train[~train['Industry_Status'].isin(['Not in universe or children'])]
test = test[~test['Industry_Status'].isin(['Not in universe or children'])]

# 입력과 출력 정의
X = train.drop(columns=['Income'],axis=1)
y = train['Income']

X = pd.get_dummies(X, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

df = X 
df.columns = df.columns.to_series().apply(lambda x: x.replace('[', '').replace(']', '').replace('<', '')).astype(str)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= random_state )

print(X.dtypes)
X = X.fillna(0)
test = test.fillna(0)

# 학습 데이터와 테스트 데이터의 열을 맞추기
missing_cols = set(X.columns) - set(test.columns)
for c in missing_cols:
    test[c] = 0
test = test[X.columns]

class EnsembleRegressor:
    def __init__(self, xgb_params, catboost_params,
                 # +lgbm_params,
                 xgb_weight, catboost_weight , 
                 # lgbm_weight
                 ):
        self.xgb_model = XGBRegressor(**xgb_params, random_state = xgb_randomstate, enable_categorical=True)
        self.catboost_model = CatBoostRegressor(**catboost_params, random_state = cat_randomstate, verbose=1)
        # self.lgbm_model = CatBoostRegressor(**lgbm_params, random_state = lgbm_randomstate, verbose=1)
        self.xgb_weight = xgb_weight
        self.catboost_weight = catboost_weight
        # self.lgbm_weight = lgbm_weight

    def fit(self, X, y):
        self.xgb_model.fit(X, y)
        self.catboost_model.fit(X, y)

    def predict(self, X):
        xgb_pred = self.xgb_model.predict(X)
        catboost_pred = self.catboost_model.predict(X)
        # lgbm_pred = self.lgbm_model.predict(X)
        
        # 가중 평균 계산
        return (self.xgb_weight * xgb_pred + self.catboost_weight * catboost_pred # + self.lgbm_weight * lgbm_pred
                ) / (self.xgb_weight + self.catboost_weight # + self.lgbm_weight
       )


""" def objective(trial):
    xgb_params = {
        # 기존 파라미터 유지하며 세밀한 조정
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000   ),
        'max_depth': trial.suggest_int('xgb_max_depth', 1 , 100 ),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.00001, 0.3, log = True ),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 20),
        'gamma': trial.suggest_float('xgb_gamma', 0.1, 1.0, log = True ),
        'subsample': trial.suggest_float('xgb_subsample', 0.7, 1.0, log = True),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 0.9, log = True ),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0, 0.01  ),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.01, 2 ,log = True ),
        'max_delta_step': trial.suggest_int('xgb_max_delta_step', 1 , 3 , log = True),
        # 새로운 파라미터 추가
        'eval_metric': trial.suggest_categorical('xgb_eval_metric', ['rmse']),
        'objective': 'reg:squarederror',
        # 'early_stopping_rounds': 10,
    }

    catboost_params = {
        'iterations': trial.suggest_int('catboost_iterations', 100, 1000  ),
        'depth': trial.suggest_int('catboost_depth', 1, 16),
        'learning_rate': trial.suggest_float('catboost_learning_rate', 0.00001, 0.3 , log = True),
        # 'l2_leaf_reg': trial.suggest_float('catboost_l2_leaf_reg', 3, 20),
        'border_count': trial.suggest_int('catboost_border_count', 30, 255  ),
        'bagging_temperature': trial.suggest_float('catboost_bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('catboost_random_strength', 0 , 10),
        'one_hot_max_size': trial.suggest_int('catboost_one_hot_max_size', 2, 10),
        # 새로운 파라미터 추가
        'task_type': 'GPU',  # GPU 사용 설정, GPU가 없을 경우 'CPU'로 변경
        'early_stopping_rounds': 10,
        # 'reduce_learning_rate' : True,
        
    } """


def objective(trial):
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 200, 800),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0, 0.1),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.1, 2, log=True),
        'eval_metric': trial.suggest_categorical('xgb_eval_metric', ['rmse', 'mae']),
        'objective': 'reg:squarederror',
    }
    
    catboost_params = {
        'iterations': trial.suggest_int('catboost_iterations', 500, 1500),
        'depth': trial.suggest_int('catboost_depth', 4, 10),
        'learning_rate': trial.suggest_float('catboost_learning_rate', 0.01, 0.3, log=True),
        'border_count': trial.suggest_int('catboost_border_count', 64, 255),
        'bagging_temperature': trial.suggest_float('catboost_bagging_temperature', 0.0, 0.8),
        'random_strength': trial.suggest_float('catboost_random_strength', 0, 5),
        'one_hot_max_size': trial.suggest_int('catboost_one_hot_max_size', 2, 10),
        'task_type': 'GPU',  # GPU 사용 설정, GPU가 없을 경우 'CPU'로 변경
        'early_stopping_rounds': 20,
        'boosting_type': 'Ordered',  # Plain 대신 Ordered 사용
    }


    xgb_weight = trial.suggest_float('xgb_weight', 0, 1)
    catboost_weight = 1 - xgb_weight  # 두 모델의 가중치 합이 1이 되도록 설정
    # lgbm_weight = 
    model = EnsembleRegressor(xgb_params=xgb_params, catboost_params=catboost_params, # lgbm_params=lgbm_params, 
                              xgb_weight=xgb_weight, catboost_weight=catboost_weight, # lgbm_weight = lgbm_weight
                              )
 
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

    # return objective

    # model = EnsembleRegressor(xgb_params=xgb_params, catboost_params=catboost_params, xgb_weight=xgb_weight, catboost_weight=catboost_weight)
    
    # # model = EnsembleRegressor(xgb_params=xgb_params, catboost_params=catboost_params)
    # model.fit(x_train, y_train)
    # preds = model.predict(x_test)
    # rmse = np.sqrt(mean_squared_error(y_test, preds))
    # return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100 ) #timeout=600)  # 100회 시도하거나, 총 600초가 경과하면 종료

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# 최적의 하이퍼파라미터를 사용하여 모델을 다시 훈련시키고, 테스트 데이터에 대한 예측을 수행합니다.
best_xgb_params = {k.replace('xgb_', ''): v for k, v in study.best_trial.params.items() if k.startswith('xgb_')}
best_catboost_params = {k.replace('catboost_', ''): v for k, v in study.best_trial.params.items() if k.startswith('catboost_')}
best_xgb_weight = study.best_trial.params.get('xgb_weight')
best_catboost_weight = 1 - best_xgb_weight

best_model = EnsembleRegressor(xgb_params=best_xgb_params, catboost_params=best_catboost_params, xgb_weight=best_xgb_weight, catboost_weight=best_catboost_weight)
best_model.fit(X, y)  # 전체 데이터셋을 사용하여 최종 모델 훈련

# 최적화된 모델로 테스트 데이터에 대한 예측 수행
ensemble_pred = best_model.predict(x_test)

ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
print(f'Optimized Ensemble RMSE: {ensemble_rmse}')

# 이상치 제거 후의 데이터를 사용하여 모델 학습
best_model.fit(X, y)

# 학습에 사용된 특성 열 추출
features = X.columns

# 테스트 데이터에서 학습에 사용된 특성만 선택
test = test[features]

# 이상치를 제거한 테스트 데이터의 누락된 값을 0으로 대체
test = test.fillna(0)

# 테스트 데이터에 대한 예측 수행
test_preds = best_model.predict(test)

# 제출 파일 생성
submission['Income'] = test_preds
dt = datetime.datetime.now()
submission.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_rmse_{ensemble_rmse:4}.csv",index=False)

print('가중치 저장')
best_model_weights_output = os.path.join(path, 'model_{}_{}_final_weights2.h5'.format(EnsembleRegressor, 'ensemble_model'))
best_model.save_weights(best_model_weights_output)

print("저장된 가중치 명: {}".format(best_model_weights_output))

# ID : 고객의 지정 넘버로 별다른 정보가 없기 때문에 분석에 사용하지 않을 것임
# Age : 고객의 나이, 연속형 변수
# Gender : 성별, 이산형 변수
# Education_Status : 최종학력을 의미한다. 이산형 변수
# Employment_Status : 취업 상태를 의미한다. 이산형 변수
# Working_Week (Yearly) : 주당 일하는 시간으로 해석된다. 연속형변수 or 범주를 잡아 이산형으로 파악 가능
# Industry_Status : 산업분야, 이산형 변수
# Occupation_Status : 직업 분야, 이산형 변수
# Race : 인종, 이산형변수
# Hispanic_Origin : 히스패닉 출신, 이산형 변수
# Martial_Status : 결혼 여부, 이산형 변수
# Household_Status : 가족 구성, 이산형 변수
# Household_Summary : 가족 구성 요약, 이산형 변수
# Citizenship : 시민권, 이산형 변수
# Birth_Country : 국적, 이산형변수
# Birth_Country (Father) : 아버지의 국적, 이산형변수
# Birth_Country (Mother) : 어머니의 국적, 이산형변수
# Tax_Status : 세금 여부, 이산형변수
# Gains : 이득또는 매출로 보인다, 연속형 변수
# Losses : 지출로 보인다, 연속형 변수
# Dividends : 배당금, 연속형 변수
# Income_Status : 소득 상태, 이산형변수
# income : 소득, 연속형 변수
