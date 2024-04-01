import dacon.RFoptuna as RFoptuna
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
import lightgbm as lgb



# 데이터 로드
path = 'C:\_data\dacon\소득\\'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv')


# 입력과 출력 정의
X = train.drop(columns=['Income'])
y = train['Income']

X = pd.get_dummies(X, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

df = X 
df.columns = df.columns.to_series().apply(lambda x: x.replace('[', '').replace(']', '').replace('<', '')).astype(str)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X.dtypes)
X = X.fillna(0)
test = test.fillna(0)

# 학습 데이터와 테스트 데이터의 열을 맞추기
missing_cols = set(X.columns) - set(test.columns)
for c in missing_cols:
    test[c] = 0
test = test[X.columns]


class EnsembleRegressor:
    def __init__(self, xgb_params, catboost_params, xgb_weight, catboost_weight):
        self.xgb_model = XGBRegressor(**xgb_params, random_state=42, enable_categorical=True)
        self.catboost_model = CatBoostRegressor(**catboost_params, random_state=42, verbose=1)
        self.xgb_weight = xgb_weight
        self.catboost_weight = catboost_weight

    def fit(self, X, y):
        self.xgb_model.fit(X, y)
        self.catboost_model.fit(X, y)

    def predict(self, X):
        xgb_pred = self.xgb_model.predict(X)
        catboost_pred = self.catboost_model.predict(X)
        # 가중 평균 계산
        return (self.xgb_weight * xgb_pred + self.catboost_weight * catboost_pred) / (self.xgb_weight + self.catboost_weight)


def objective(trial):
    xgb_params = {
        # 기존 파라미터 유지하며 세밀한 조정
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1200, 50),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 20 ),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log = True ),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 20),
        'gamma': trial.suggest_float('xgb_gamma', 0.1, 1.0,log = True ),
        'subsample': trial.suggest_float('xgb_subsample', 0.7, 1.0, log = True),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.7, 0.9,log = True ),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0, 0.01  ),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.01, 2 ,log = True ),
        'max_delta_step': trial.suggest_int('xgb_max_delta_step', 1 , 3 , log = True),
        # 새로운 파라미터 추가
        'eval_metric': trial.suggest_categorical('xgb_eval_metric', ['rmse', 'mae']),
        'objective': 'reg:squarederror',
        # 'early_stopping_rounds': 10,
        
    }
    
    catboost_params = {
        'iterations': trial.suggest_int('catboost_iterations', 100, 1200 , 50),
        'depth': trial.suggest_int('catboost_depth', 4, 20),
        'learning_rate': trial.suggest_float('catboost_learning_rate', 0.01, 0.3 , log = True),
        # 'l2_leaf_reg': trial.suggest_float('catboost_l2_leaf_reg', 3, 20),
        'border_count': trial.suggest_int('catboost_border_count', 30, 255  ),
        'bagging_temperature': trial.suggest_float('catboost_bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('catboost_random_strength', 0 , 10),
        'one_hot_max_size': trial.suggest_int('catboost_one_hot_max_size', 2, 10),
        # 새로운 파라미터 추가
        'task_type': 'GPU',  # GPU 사용 설정, GPU가 없을 경우 'CPU'로 변경
        'early_stopping_rounds': 50,
        # 'reduce_learning_rate' : True,
    
    }
    
    lgbm_params = {
    # 기존 파라미터 유지하며 세밀한 조정
    'n_estimators': trial.suggest_int('lgbm_n_estimators', 100, 1200, 50),
    'max_depth': trial.suggest_int('lgbm_max_depth', 3, 20),
    'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.3, log=True),
    'min_child_weight': trial.suggest_int('lgbm_min_child_weight', 1, 20),
    'subsample': trial.suggest_float('lgbm_subsample', 0.7, 1.0, log=True),
    'colsample_bytree': trial.suggest_float('lgbm_colsample_bytree', 0.7, 0.9, log=True),
    'reg_alpha': trial.suggest_float('lgbm_reg_alpha', 0, 0.01),
    'reg_lambda': trial.suggest_float('lgbm_reg_lambda', 0.01, 2, log=True),
    'max_delta_step': trial.suggest_int('lgbm_max_delta_step', 1, 3, log=True),
    # 새로운 파라미터 추가
    'objective': 'regression',
    'metric': 'rmse',  # 또는 'mae'를 사용할 수 있습니다.
    'boosting_type': 'gbdt',  # 또는 'dart' 또는 'goss'
    # 'early_stopping_round': 10,
    # 'feature_fraction_bynode': 0.8,
}
    xgb_weight = trial.suggest_float('xgb_weight', 0, 1)
    catboost_weight = 1 - xgb_weight  # 두 모델의 가중치 합이 1이 되도록 설정
    model = EnsembleRegressor(xgb_params=xgb_params, catboost_params=catboost_params, xgb_weight=xgb_weight, catboost_weight=catboost_weight)

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

study = RFoptuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100) #timeout=600)  # 100회 시도하거나, 총 600초가 경과하면 종료

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

# 테스트 데이터에 대한 예측 수행
test_preds = best_model.predict(test)
submission['Income'] = test_preds

# 제출 파일 생성
dt = datetime.datetime.now()
submission.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}_rmse_{ensemble_rmse:4}.csv",index=False)