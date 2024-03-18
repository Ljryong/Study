
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
import tensorflow as tf
import pandas as pd
import numpy as np
import optuna
import random

RANDOM_STATE = 42  
tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

train_csv = pd.read_csv('C:\_data\dacon\RF/train.csv',index_col=0)

x = train_csv.drop('login',axis=1)
y = train_csv['login']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=RANDOM_STATE)

def objectiveRF(trial):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 100, 1000, 10),
        'criterion' : trial.suggest_categorical('criterion', ['gini','entropy']),
        'bootstrap' : trial.suggest_categorical('bootstrap', [True,False]),
        'max_depth' : trial.suggest_int('max_depth', 5, 100 ),
        'random_state' : RANDOM_STATE,
        'min_samples_split' : trial.suggest_int('min_samples_split', 2, 100),
        'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 100),
        # 'min_samples_split' : trial.suggest_uniform('min_samples_split',0,1), 
        # 'min_samples_leaf' : trial.suggest_uniform('min_samples_leaf',0,0.5), 
        'min_weight_fraction_leaf' : trial.suggest_uniform('min_weight_fraction_leaf',0,0.5),
    }
    
    # 학습 모델 생성
    model = RandomForestClassifier(**param)
    rf_model = model.fit(x_train, y_train) # 학습 진행
    
    # 모델 성능 확인
    # score = auc(rf_model.predict(x_test), y_test)
    score = model.score(x_test,y_test)
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objectiveRF, n_trials=1000)

best_params = study.best_params
print(best_params)

optuna.visualization.plot_param_importances(study)      # 파라미터 중요도 확인 그래프
optuna.visualization.plot_optimization_history(study)   # 최적화 과정 시각화

# model = RandomForestClassifier(**best_params)
model = RandomForestClassifier(**best_params)
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
pred_list = model.predict_proba(x_test)[:,1]
print("score: ",score)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test,pred_list)
print("AUC:  ",auc)

#  Trial 99 finished with value: 0.9007633587786259 and parameters: {'n_estimators': 879, 'criterion': 'gini', 'bootstrap': False, 'max_depth': 29, 'min_samples_split': 0.19835986327450797, 'min_samples_leaf': 0.2229979795940108, 'min_weight_fraction_leaf': 0.3422843335377289}. Best is trial 89 with value: 0.9122137404580153.
# {'n_estimators': 662, 'criterion': 'gini', 'bootstrap': False, 'max_depth': 32, 'min_samples_split': 0.3746056675007304, 'min_samples_leaf': 0.029695340582856743, 'min_weight_fraction_leaf': 0.04368839461045142}

submit_csv = pd.read_csv('C:\_data\dacon\RF/sample_submission.csv')
for label in submit_csv:
    if label in best_params.keys():
        submit_csv[label] = best_params[label]
    
submit_csv.to_csv(f'C:\_data\dacon\RF/submit_AUC_{auc:.6f}.csv',index=False)


#  {'n_estimators': 620, 'criterion': 'gini', 'bootstrap': False, 'max_depth': 89, 'min_samples_split': 0.03410033774641412, 'min_samples_leaf': 0.015484760122027784, 'min_weight_fraction_leaf': 0.008924552087879787}. Best is trial 2063 with value: 0.9695431472081218.
# {'n_estimators': 600, 'criterion': 'gini', 'bootstrap': True, 'max_depth': 94, 'min_samples_split': 0.014404147547730842, 'min_samples_leaf': 0.00017914895480158378, 'min_weight_fraction_leaf': 5.026888393899927e-05}
# score:  0.9644670050761421
# AUC:   0.8515193370165747


# [I 2024-03-14 22:01:52,621] Trial 9999 finished with value: 0.9593908629441624 and parameters: {'n_estimators': 810, 'criterion': 'entropy', 'bootstrap': True, 'max_depth': 75, 'min_samples_split': 0.04785613352733973, 'min_samples_leaf': 0.00030107321429915547, 'min_weight_fraction_leaf': 0.008798754793372283}. Best is trial 495 with value: 0.9644670050761421.
# {'n_estimators': 770, 'criterion': 'entropy', 'bootstrap': True, 'max_depth': 85, 'min_samples_split': 0.0007376062906313298, 'min_samples_leaf': 2.6548309394764302e-05, 'min_weight_fraction_leaf': 0.018105039249163634}
# score:  0.9644670050761421
# AUC:   0.8988259668508287