import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import random

for step in range(100) : 
    RAMDOM = random.randint(0,10000)
    random_state    = RAMDOM
    xgb_random = RAMDOM
    cat_random = RAMDOM
    kfold_random = RAMDOM

    path = 'C:\_data\dacon\소득\\'
    train_csv = pd.read_csv(path + "train.csv", index_col=0)
    test_csv = pd.read_csv(path + "test.csv", index_col=0)

    train_csv = train_csv.drop(['Gains', 'Losses','Dividends'], axis=1)
    test_csv = test_csv.drop(['Gains', 'Losses', 'Dividends'], axis=1)

    # print(pd.value_counts(train_csv['Industry_Status']))

    # Age < 83 이상치 제거
    # Gains < 10 제거
    # Losses < 50 제거 
    # Income < 2200 제거 
    # Dividends < 1 제거 했을떄 안했을 떄 확인 필요

    # Income_Status Unknown 제거 생각

    # Age	Gender	Education_Status	Employment_Status	Working_Week (Yearly)	Industry_Status	Occupation_Status	
    # Race	Hispanic_Origin	Martial_Status	Household_Status	Household_Summary	Citizenship	Birth_Country	Birth_Country (Father)	
    # Birth_Country (Mother)	Tax_Status	Gains	Losses	Dividends	Income_Status	Income

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
    train_csv['Household_Status'] = train_csv['Household_Status'].map(category_mapping)
    test_csv['Household_Status'] = test_csv['Household_Status'].map(category_mapping)

    """ 
    # Armed Forces가 포함된 행 삭제
    # train_csv = train_csv[train_csv['Occupation_Status'] != 'Armed Forces']
    # test_csv = test_csv[test_csv['Occupation_Status'] != 'Armed Forces']

    # Industry_Status
    # train_csv = train_csv[(train_csv['Industry_Status'] != 'Armed Forces') & (train_csv['Industry_Status'] != 'Forestry & Fisheries')]
    # test_csv = test_csv[(test_csv['Industry_Status'] != 'Armed Forces') & (test_csv['Industry_Status'] != 'Forestry & Fisheries')]
    """

    train_csv.loc[train_csv['Occupation_Status'] == 'Armed Forces', 'Occupation_Status'] = 'Unknown'
    test_csv.loc[test_csv['Occupation_Status'] == 'Armed Forces', 'Occupation_Status'] = 'Unknown'

    train_csv.loc[train_csv['Industry_Status'] == 'Armed Forces', 'Industry_Status'] = 'Unknown'
    test_csv.loc[test_csv['Industry_Status'] == 'Armed Forces', 'Industry_Status'] = 'Unknown'
    train_csv.loc[train_csv['Industry_Status'] == 'Forestry & Fisheries', 'Industry_Status'] = 'Unknown'
    test_csv.loc[test_csv['Industry_Status'] == 'Forestry & Fisheries', 'Industry_Status'] = 'Unknown'

    categorical_features = train_csv.select_dtypes(include='object').columns.values

    lbe = LabelEncoder()
    for feature in categorical_features:
        lbe.fit(train_csv[feature])
        train_csv[feature] = lbe.transform(train_csv[feature])
        for label in test_csv[feature]:
            if label not in lbe.classes_:
                lbe.classes_ = np.append(lbe.classes_, label)
        test_csv[feature] = lbe.transform(test_csv[feature])

    X = train_csv.drop(["Income"], axis=1)
    y = train_csv['Income']

    # 결측치 제거
    X = X.interpolate(method='linear', limit_direction='forward', axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # 스케일링이 필요한 열 선택
    columns_to_scale = ['Age', 'Working_Week (Yearly)']

    # StandardScaler 객체 생성
    scaler = StandardScaler()

    # 스케일링을 적용할 데이터 선택
    data_to_scale = X_train[columns_to_scale]
    data_to_scale2 = X_test[columns_to_scale]

    # 선택된 열들에 대해 스케일링 수행
    scaled_data = scaler.fit_transform(data_to_scale)
    scaled_data2 = scaler.transform(data_to_scale2)

    # 스케일링된 데이터를 다시 원본 데이터프레임에 삽입
    X_train[columns_to_scale] = scaled_data
    X_test[columns_to_scale] = scaled_data2

    kfold = KFold(n_splits=5, shuffle=True, random_state=kfold_random)

    def objective(trial):
        # catboost_params = {
        #     'iterations': trial.suggest_int('iterations', 500, 1500),
        #     'depth': trial.suggest_int('depth', 4, 10),
        #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        #     'border_count': trial.suggest_int('border_count', 64, 255),
        #     'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 0.8),
        #     'random_strength': trial.suggest_float('random_strength', 0, 5),
        #     'task_type': 'GPU',  # GPU 사용 설정, GPU가 없을 경우 'CPU'로 변경
        #     'early_stopping_rounds': 20,
        #     'boosting_type': 'Ordered',  # Plain 대신 Ordered 사용
        #     'silent': True
        # }
        
        
        catboost_params = {
        'iterations': trial.suggest_int('iterations', 500, 1500),
        'depth': trial.suggest_int('depth', 4, 16),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        # 'border_count': trial.suggest_int('border_count', 64, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 0.8),
        'random_strength': trial.suggest_float('random_strength', 0, 5),
        'task_type': 'GPU',  # GPU 사용 설정, GPU가 없을 경우 'CPU'로 변경
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 5, 50),
        'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
        'silent': True
    }
        
        cat = CatBoostRegressor(**catboost_params , random_state=cat_random )
        
        rmse_scores = []
        for train_idx, val_idx in kfold.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            cat.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], early_stopping_rounds=20, verbose=False)
            
            pred = cat.predict(X_val_fold)
            rmse = np.sqrt(mean_squared_error(pred, y_val_fold))
            rmse_scores.append(rmse)
        
        return np.mean(rmse_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials= 100 )

    best_params = study.best_params
    best_rmse = study.best_value

    print("Best RMSE:", best_rmse)
    print("Best parameters:", best_params)

    cat = CatBoostRegressor(**best_params, random_state=cat_random ,eval_metric = 'RMSE' )
    cat.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)], early_stopping_rounds=20, verbose=False  )

    pred = cat.predict(X_test)
    final_rmse = np.sqrt(mean_squared_error(pred, y_test))
    print(f'Final RMSE: {final_rmse}')

    import time
    timestr = time.strftime("%Y%m%d%H%M%S")
    save_name = timestr

    submission_csv = pd.read_csv(path + "sample_submission.csv")
    predictions = cat.predict(test_csv)
    submission_csv["Income"] = predictions
    
    if final_rmse < 585:
        submission_csv.to_csv(path + f"sample_submission_pred{save_name}_{final_rmse}_random_{RAMDOM}.csv", index=False)
    else:
        print("RMSE is not less than 580. Model not saved.")
    
    
