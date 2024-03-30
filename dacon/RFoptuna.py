import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import optuna
from sklearn.metrics import roc_auc_score

SEED1 = 42
SEED2 = 42
SEED3 = 42

data = pd.read_csv('C:\_data\dacon\RF\\train.csv', index_col=0)
submit = pd.read_csv('C:\_data\dacon\RF\sample_submission.csv')

# person_id 컬럼 제거
X_train = data.drop(['login'],axis=1)
Y_train = data['login']

x_train , x_test , y_train , y_test = train_test_split(X_train,Y_train,test_size = 0.2 , random_state=42 , stratify=Y_train )

def objective(trial):
    # param_space = {
    #     'n_estimators': trial.suggest_int('n_estimators', 1, 100),
    #     'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
    #     'max_depth': trial.suggest_int('max_depth', 1, 100 ),
    #     'min_samples_split': trial.suggest_int('min_samples_split', 2, 100 ),
    #     'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50 ),
    #     'max_features': trial.suggest_categorical('max_features', ['auto']),   # auto
    #     'bootstrap': trial.suggest_categorical('bootstrap', [True]),
    # }
    
    param_space = {
    'n_estimators': trial.suggest_int('n_estimators', 50, 150),
    'criterion': trial.suggest_categorical('criterion', ['gini','entropy']),
    'max_depth': trial.suggest_int('max_depth', 1, 100),
    'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),
    'min_weight_fraction_leaf': 0.0,
    'max_features': trial.suggest_categorical('max_features', ['auto']),       # 'sqrt', 'log2', None,
    'max_leaf_nodes': None, 
    # 'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5),
    'bootstrap': trial.suggest_categorical('bootstrap', [True]),
}
    
    
    model = RandomForestClassifier(**param_space , random_state = 42)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)

best_params = study.best_params
best_score = study.best_value

print("Best Params:", best_params)
print("Best Score:", best_score)

# 찾은 최적의 파라미터들을 제출 양식에 맞게 제출
for param, value in best_params.items():
    if param in submit.columns:
        submit[param] = value




import datetime
dt = datetime.datetime.now()
submit.to_csv(f'C:/_data/dacon/RF/submit_{dt.day}day{dt.hour:2}{dt.minute:2}_AUC_{best_score:.4f}.csv',index=False)
