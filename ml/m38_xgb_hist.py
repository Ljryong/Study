from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer , load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold , StratifiedGroupKFold
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler , StandardScaler

#1 데이터
# x, y = load_breast_cancer(return_X_y=True)
x, y = load_diabetes(return_X_y=True)

x_train , x_test , y_train , y_test = train_test_split(x,y, random_state= 777 ,test_size= 0.2 ,
                                                    #    stratify=y
                                                       )

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
# kfold = StratifiedGroupKFold(n_splits=n_splits , shuffle=True , random_state= 777 )
# kfold = KFold(n_splits=n_splits , shuffle=True , random_state= 777 )

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

parameters = {'n_estimater' :100 ,
              'learning_rate' : 0.1 ,
              'max_depth' : 6 ,
              'min_child_weight' : 10 ,
              }

#2 모델
model = XGBRegressor()                  # sklearn에 rapping 되어 있음
# model = XGBClassifier()

#3 훈련
model.set_params(**parameters , early_stopping_rounds = 20 , random_state = 777  )
model.fit(x_train,y_train , eval_set = [(x_train,y_train),(x_test,y_test)] , # eval_set = validation
          verbose = 1,          # verbose = True 가 디폴트
          eval_metric = 'mae' ,    # default = rmse // mse는 없다 error // mae 가능 // mape(mean absolute percentage error) //  rmsle
                                   # error 2진분류에서 통상 사용(acc랑 다른 정확도가 나옴) // logloss 이진분류 디폴트, 다중분류 X (acc랑 다른 정확도가 나옴)// 
                                   # auc 이진 다중 전부 먹힘 하지만 이진에서가 좋음(acc랑 다른 정확도가 나옴) // 
                                   # mlogloss 다중분류 디폴트 (acc랑 다른 정확도가 나옴) // merror 다중분류(acc랑 다른 정확도가 나옴) // 
                                   # auc 이진 다중 전부 먹힘 하지만 이진에서가 좋음 (acc랑 다른 정확도가 나옴)
          )            

#4 평가
result = model.score(x_test,y_test)
print('최종 점수4 : ' , result )

y_predict = model.predict(x_test)
from sklearn.metrics import accuracy_score , roc_auc_score , f1_score , r2_score , mean_absolute_error
# print('acc : ' , accuracy_score(y_test,y_predict))
# print('acc : ' , f1_score(y_test,y_predict))
# print('acc : ' , roc_auc_score(y_test,y_predict))
print('r2 : ' , r2_score(y_test,y_predict))
print('mae : ' , mean_absolute_error(y_test,y_predict))


print('================================')
hist = model.evals_result()             # early stopping 지점 찾기
print(hist)
import matplotlib.pyplot as plt
# item() = 딕셔너리 형태를 받아서 튜플 형태로 반환해줌
# 평가 지표 플롯 그리기                 
plt.figure(figsize=(10, 6))
train_mae = hist['validation_0']['mae']     # Train 데이터 평가 결과
val_mae = hist['validation_1']['mae']       # Validation 데이터 평가 결과
plt.plot(train_mae , label = 'train' )
plt.plot(val_mae, label = 'val')
plt.xlabel('Epochs')
plt.ylabel('Metric')
plt.title('Training and Validation Metrics')
plt.grid()
plt.legend()
plt.show()



# 최종 점수 :  0.19902639314080917
# 최종 점수2 :  0.21239130913036297
# 최종 점수3 :  0.3258849470312424
# 최종 점수4 :  0.3295508353786477



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

