import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import optuna

random_state = 42

# 데이터 불러오기
path = 'C:\_data\dacon\소득\\'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)

# 이상치 처리 및 전처리 (생략)
train = train.drop(['Gains', 'Losses','Dividends'], axis=1)
test = test.drop(['Gains', 'Losses', 'Dividends'], axis=1)

# 입력과 출력 정의
X = train.drop(columns=['Income'], axis=1)
y = train['Income']

# 데이터 전처리 (생략)
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


# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Objective 함수 정의
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1500),
        'max_depth': trial.suggest_categorical('max_depth', [ None,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 1.0 , ),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.001, 1.0),
        'random_strength': trial.suggest_loguniform('random_strength', 1e-9, 10.0),
        'task_type': 'GPU',  # GPU 사용 설정, GPU가 없을 경우 'CPU'로 변경
        'early_stopping_rounds': 10,
        # 다른 하이퍼파라미터도 필요에 따라 추가 가능
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse

# Study 생성 및 최적화
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# 최적의 하이퍼파라미터로 최종 모델 훈련
best_params = study.best_trial.params
best_model = CatBoostRegressor(**best_params , random_state=random_state)
best_model.fit(X, y)  # 전체 데이터셋을 사용하여 최종 모델 훈련

# 이하 예측 및 결과 저장 코드 (생략)
print('best params : ' , best_params)
# 테스트 데이터에 대한 예측 수행
test_preds = best_model.predict(test)

# 제출 파일 생성
import datetime
submission = pd.read_csv(path + 'sample_submission.csv' )
submission['Income'] = test_preds
dt = datetime.datetime.now()
submission.to_csv(path+f"submit_{dt.day}day{dt.hour:2}{dt.minute:2}.csv",index=False)

# 1756
# best params :  {'iterations': 615, 'max_depth': 16, 'learning_rate': 0.16733858163949164, 'l2_leaf_reg': 9.992161762588143,
#                 'border_count': 210, 'bagging_temperature': 0.31713163747124323, 'random_strength': 9.636336842651728e-09}

