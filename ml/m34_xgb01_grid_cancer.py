from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold , StratifiedGroupKFold
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler , StandardScaler

#1 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train , x_test , y_train , y_test = train_test_split(x,y, random_state= 777 ,test_size= 0.2 ,
                                                       stratify=y
                                                       )

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedGroupKFold(n_splits=n_splits , shuffle=True , random_state= 777 )
kfold = KFold(n_splits=n_splits , shuffle=True , random_state= 777 )

'''
'n_estimater' : [100,200,300,400,500,1000] # 디폴트 100 / 1~inf / 정수
'learning_rate' : [0.1,0.2,0.3,0.5,1,0.01,0.001] # 디폴트 0.3 / 0~1 / eta 제일 중요  
# learning_rate(훈련율) : 작을수록 디테일하게 보고 크면 클수록 듬성듬성 본다. batch_size랑 비슷한 느낌
#                        하지만 너무 작으면 오래 걸림 데이터의 따라 잘 조절 해야된다
'max_depth' : [None,2,3,4,5,6,7,8,9,10] # 디폴트 6 / 0~inf / 정수    # tree의 깊이를 나타냄
'gamma' : [0,1,2,3,4,5,7,10,100] # 디폴트 0 / 0~inf 
'min_child_weight' : [0,0.01,0.001,0.1,0.5,1,5,10,100] # 디폴트 1 / 0~inf 
'subsample' : [0,0.1,0.2,0.3,0.5,0.7,1] # 디폴트 1 / 0~1
'colsample_bytree' : [0,0.1,0.2,0.3,0.5,0.7,1] # 디폴트 1 / 0~1 
'colsample_bylevel' : [0,0.1,0.2,0.3,0.5,0.7,1] # 디폴트 1 / 0~1
'colsample_bynode' : [0,0.1,0.2,0.3,0.5,0.7,1] # 디폴트 1 / 0~1
'reg_alpha' : [0,0.1,0.01,0.001,1,2,10] # 디폴트 0 / 0~inf / L1 절대값 가중치 규제 / alpha / 중요
'reg_lambda' : [0,0.1,0.01,0.001,1,2,10] # 디폴트 1 / 0~inf / L2 제곱 가중치 규제 / lambda / 중요

'''

parameters = {'n_estimater' :[100],
              'learning_rate' : [0.1,0.2,0.3,0.5,1,0.01,0.001],
              'max_depth' : [3],
}

#2 모델
xgb = XGBClassifier(random_state = 777)
model = RandomizedSearchCV(xgb , parameters , cv = kfold , 
                           n_jobs=22,                   # cpu 사용 코어 수 //  다 쓰는 것 보다는 1~2개 여유를 주는게 좋다
                           
                           )

#3 훈련
model.fit(x_train,y_train)

#4 평가, 예측
print('최상의 매겨변수 :', model.best_estimator_ )
print('최상의 매겨변수 :', model.best_params_    )
print('최상의 점수 :',     model.best_score_     )              # 이대로 사용하게 되면 test가 아니라 train의 결과

result = model.score(x_test,y_test)

print('최종 점수 : ' , result )


'n_estimater' 
'learning_rate'  
'max_depth' 
'gamma' 
'min_child_weight' 
'subsample' 
'colsample_bytree' 
'colsample_bylevel' 
'colsample_bynode' 
'reg_alpha' 
'reg_lambda' 

