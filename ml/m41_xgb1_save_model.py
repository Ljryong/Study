from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer , load_diabetes , load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler , StandardScaler
import xgboost as xgb
print(xgb.__version__)          # 2.0.3

#1 데이터
x, y = load_digits(return_X_y=True)
x_train , x_test , y_train , y_test = train_test_split(x,y, random_state= 777 ,test_size= 0.2 ,
                                                       stratify=y
                                                       )

# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimater' :100 ,
              'learning_rate' : 0.1 ,
              'max_depth' : 6 ,
              'min_child_weight' : 10 ,
              }

#2 모델
model = XGBClassifier()

#3 훈련
model.set_params(**parameters , early_stopping_rounds = 20 , random_state = 777  )
model.fit(x_train,y_train , eval_set = [(x_train,y_train),(x_test,y_test)] , # eval_set = validation
          verbose = 1,          # verbose = True(1) 가 디폴트
          )

#4 평가
result = model.score(x_test,y_test)
print('최종 점수4 : ' , result )

y_predict = model.predict(x_test)
from sklearn.metrics import accuracy_score , roc_auc_score , f1_score , r2_score
print('acc : ' , accuracy_score(y_test,y_predict))

import pickle
# 대부분의 경우 작은 크기의 데이터를 저장할 때는 pickle을 사용하는 것이 간편하고 효과적일 수 있지만, 큰 배열과 같은 대용량 데이터를 저장할 때는 joblib이 더 효율적일 수 있습니다.
# path = 'C:/_data/_save/_pickle_test//'
# pickle.dump(model, open(path + 'm39_pickle_save.dat' , 'wb' ))

import joblib
# path = 'C:/_data/_save/_joblib_test//'
# joblib.dump(model , path + 'm40_joblib_save.dat'  )

path = 'C:/_data/_save//'
model.save_model(path + 'm41_xgb1_save_model.dat'  )


# 최종 점수4 :  0.9611111111111111
# acc :  0.9611111111111111






