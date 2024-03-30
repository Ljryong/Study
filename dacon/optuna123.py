import optuna
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import numpy as np
import random as rn
from sklearn.model_selection import train_test_split

while True : 
    SEED1 = 42
    SEED2 = 42
    SEED3 = 42

    tf.random.set_seed(SEED1)  
    np.random.seed(SEED2)
    rn.seed(SEED3)

    data = pd.read_csv('C:\_data\dacon\RF\\train.csv', index_col=0)
    submit = pd.read_csv('C:\_data\dacon\RF\sample_submission.csv')

    # person_id 컬럼 제거
    X_train = data.drop(['login'],axis=1)
    Y_train = data['login']

    x_train , x_test , y_train , y_test = train_test_split(X_train,Y_train,test_size = 0.18 , random_state=42 , stratify=Y_train )


    def objective(trial):
        # params = {
        #     'n_estimators': trial.suggest_int('n_estimators', 1, 50, ),
        #     'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        #     'max_depth': trial.suggest_int('max_depth',1 , 30  ),
        #     'min_samples_split': trial.suggest_int('min_samples_split', 2, 30 ),
        #     'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1,  10),
        #     'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        #     'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        # }
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 50 ),
            'criterion': trial.suggest_categorical('criterion', ['gini','entropy']),
            'max_depth': trial.suggest_int('max_depth', 1, 100),
            'min_samples_split': trial.suggest_int('min_samples_split', 2,100 ),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),
            'min_weight_fraction_leaf': 0.0,
            'max_features': trial.suggest_categorical('max_features', ['auto']),       # 'sqrt', 'log2', None,
            'max_leaf_nodes': None, 
            # 'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5), = 0
            'bootstrap': trial.suggest_categorical('bootstrap', [True]),
        }
        
        # 학습 모델 생성
        model = RandomForestClassifier(**params, random_state=42)
        rf_model = model.fit(x_train, y_train) # 학습 진행
        
        # 모델 성능 확인
        # score = auc(rf_model.predict(x_test), y_test)
        score = model.score(x_test,y_test)
        
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    print(best_params)

    model = RandomForestClassifier(**best_params)
    model.fit(x_train,y_train)
    score = model.score(x_test,y_test)
    pred_list = model.predict_proba(x_test)[:,1]
    print("score: ",score)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test,pred_list)
    print("AUC:  ",auc)

    if auc > 0.855 :
        submit_csv = pd.read_csv('C:\_data\dacon\RF\sample_submission.csv')
        for label in submit_csv:
            if label in best_params.keys():
                submit_csv[label] = best_params[label]
        submit_csv.to_csv(f'C:\_data\dacon\RF\submit_AUC_{auc:.6f}_1st_JR.csv',index=False)




