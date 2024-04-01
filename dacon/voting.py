# 필요한 라이브러리 import
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
import optuna
import time

start = time.time()

cat_random = 220118
lgbm_random = 220118
xgb_random = 220118
train_test_random = 220118

# 함수 정의: 이상치 처리
def outliers(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return (data > upper_bound) | (data < lower_bound)

# 데이터 경로 및 시드 설정
path = 'C:/_data/dacon/소득/'

# 데이터 읽기
train_df = pd.read_csv(path + "train.csv", index_col=0)
test_df = pd.read_csv(path + "test.csv", index_col=0)

# 라벨인코더 
encoding_target = list(train_df.dtypes[train_df.dtypes == "object"].index)

for i in encoding_target:
    le = LabelEncoder()
    
    # train과 test 데이터셋에서 해당 열의 모든 값을 문자열로 변환
    train_df[i] = train_df[i].astype(str)
    test_df[i] = test_df[i].astype(str)
    
    le.fit(train_df[i])
    train_df[i] = le.transform(train_df[i])
    
    # test 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
    for case in np.unique(test_df[i]):
        if case not in le.classes_: 
            le.classes_ = np.append(le.classes_, case)
    
    test_df[i] = le.transform(test_df[i])



# 이상치 제거 및 불필요한 열 삭제
train_df = train_df.drop(['Gains', 'Losses','Dividends'], axis=1)
test_df = test_df.drop(['Gains', 'Losses', 'Dividends'], axis=1)

""" # 명목형 데이터 원-핫 인코딩
nominal_columns = ['Gender', 'Race', 'Employment_Status', 'Industry_Status','Occupation_Status','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)']
one_hot_encoder = OneHotEncoder(sparse=False)
ohe_train = one_hot_encoder.fit_transform(train_df[nominal_columns])
ohe_test = one_hot_encoder.transform(test_df[nominal_columns])
nominal_train_df = pd.DataFrame(ohe_train, columns=one_hot_encoder.get_feature_names_out(nominal_columns), index=train_df.index)
nominal_test_df = pd.DataFrame(ohe_test, columns=one_hot_encoder.get_feature_names_out(nominal_columns), index=test_df.index)
"""

# 범주형 데이터 레이블 인코딩
lbe = LabelEncoder()
categorical_features = [col for col in train_df.columns if train_df[col].dtype == 'object']
for feature in categorical_features:
    lbe.fit(train_df[feature])
    train_df[feature] = lbe.transform(train_df[feature])
    test_df[feature] = lbe.transform(test_df[feature])

# 학습 및 테스트 데이터 분리
X = train_df.drop(['Income'], axis=1)
y = train_df['Income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = train_test_random)

# CatBoost 모델 최적화 함수
def optimize_catboost(trial):
    catboost_params = {
        'iterations': trial.suggest_int('iterations', 100, 1000 , 50),
        'depth': trial.suggest_int('depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.03 , log = True),
        # 'l2_leaf_reg': trial.suggest_float('catboost_l2_leaf_reg', 3, 20),
        # 'border_count': trial.suggest_int('border_count', 30, 255  ),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0 , 10),
        # 'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 10),
        # 새로운 파라미터 추가
        'task_type': 'GPU',  # GPU 사용 설정, GPU가 없을 경우 'CPU'로 변경
        'early_stopping_rounds': 10,
        # 'reduce_learning_rate' : True,
        
    }
    model = CatBoostRegressor(**catboost_params, eval_metric="RMSE", random_state=cat_random)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    predictions = model.predict(X_test)
    return np.sqrt(mean_squared_error(predictions, y_test))

# LightGBM 모델 최적화 함수
def optimize_lgbm(trial):
    lgbm_params = {
    # 기존 파라미터 유지하며 세밀한 조정
    'n_estimators': trial.suggest_int('_n_estimators', 100, 1200, 50),
    'max_depth': trial.suggest_int('max_depth', 5, 20),
    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.03, log=True),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
    'subsample': trial.suggest_float('subsample', 0.5, 1.0, log=True),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9, log=True),
    'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.01),
    'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 2, log=True),
    'max_delta_step': trial.suggest_int('max_delta_step', 1, 3, log=True),
    # 새로운 파라미터 추가
    'objective': 'regression',
    'metric': 'rmse',  # 또는 'mae'를 사용할 수 있습니다.
    'boosting_type': 'gbdt',  # 또는 'dart' 또는 'goss'
    # 'early_stopping_round': 10,
    # 'feature_fraction_bynode': 0.8,
}
    model = LGBMRegressor(**lgbm_params, random_state=lgbm_random)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)] )
    predictions = model.predict(X_test)
    return np.sqrt(mean_squared_error(predictions, y_test))

# XGBoost 모델 최적화 함수
def optimize_xgb(trial):
    xgb_params = {
        # 기존 파라미터 유지하며 세밀한 조정
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, 50),
        'max_depth': trial.suggest_int('max_depth', 5, 20 ),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.03, log = True ),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.1, 1.0,log = True ),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0, log = True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9,log = True ),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.01  ),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 2 ,log = True ),
        'max_delta_step': trial.suggest_int('max_delta_step', 1 , 3 , log = True),
        # 새로운 파라미터 추가
        'eval_metric': trial.suggest_categorical('eval_metric', ['rmse']),
        'objective': 'reg:squarederror',
        # 'early_stopping_rounds': 10,
    }
    model = XGBRegressor(**xgb_params, random_state=xgb_random)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    predictions = model.predict(X_test)
    return np.sqrt(mean_squared_error(predictions, y_test))

# Optuna를 사용하여 CatBoost, LightGBM, XGBoost 모델의 최적 하이퍼파라미터 찾기
study_catboost = optuna.create_study(study_name="CatBoost Optimization", direction="minimize")
study_catboost.optimize(optimize_catboost, n_trials=100)

study_lgbm = optuna.create_study(study_name="LightGBM Optimization", direction="minimize")
study_lgbm.optimize(optimize_lgbm, n_trials=100)

study_xgb = optuna.create_study(study_name="XGBoost Optimization", direction="minimize")
study_xgb.optimize(optimize_xgb, n_trials=100)

# 최적화된 하이퍼파라미터 가져오기
catboost_params = study_catboost.best_params
lgbm_params = study_lgbm.best_params
xgb_params = study_xgb.best_params

# CatBoost, LightGBM, XGBoost 모델 초기화 및 최적 파라미터 적용
catboost_model = CatBoostRegressor(**catboost_params, eval_metric="RMSE", random_state=cat_random)
lgbm_model = LGBMRegressor(**lgbm_params, random_state=lgbm_random)
xgb_model = XGBRegressor(**xgb_params, random_state=xgb_random)

# 보팅 모델 초기화
voting_model = VotingRegressor(
    estimators=[
        ('catboost', catboost_model),
        ('lgbm', lgbm_model),
        ('xgb', xgb_model)
    ]
)

# 보팅 모델 학습
voting_model.fit(X_train, y_train)

# 보팅 모델 예측
voting_predictions = voting_model.predict(X_test)

# 평가 및 출력
voting_score = np.sqrt(mean_squared_error(voting_predictions, y_test))
print(f'Voting RMSE: {voting_score}')

# 테스트 데이터 예측 및 제출용 CSV 생성
submission_csv = pd.read_csv(path + "sample_submission.csv")
test_predictions = voting_model.predict(test_df)
submission_csv["Income"] = test_predictions
submission_csv.to_csv(path + f"sample_submission_voting_{voting_score}.csv", index=False)

end = time.time()

print('걸린 시간 : ',end - start)