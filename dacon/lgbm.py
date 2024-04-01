import lightgbm as lgb

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
