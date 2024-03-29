# 배깅 보팅 부스팅 스태킹 ? 

# voting 모델을 여러개 사용함
# 2가지의 방식이 있고 soft와 hard가 있음 많은 투표를 받은 쪽을 선택함
# hard : 모델을 홀수로 잡아야 됨// 다수결로 여러개 중에 하나? // Reggressor 는 hard voting 을 하기 어렵다 = 안해 
# soft : 모델을 다 더해서 모델의 수 만큼 나눠서 사용

# bagging
# 모델이 1종류, 하지만 데이터가 다르다
# n_estimators = epoch 랑 다름 // n_estimators 다 다른 데이터 들로 훈련을 함 // n_estimators =  다른데이터로 n_estimators 만큼 훈련 , epoch = 같은데이터로 epoch 만큼 훈련
# n_estimators는 만약 0~9까지 있다면 내가 지정한 수 만큼을 계속 랜덤하게 뽑아내서 훈련함, 중복은 가능하지만 지정한 수가 전부 똑같게는 하지 않음
# 중복이 가능하다는 것은 0,0,0,0,0,0,0... 이런식으로도 가능하다는 뜻
# 빼는 것이 Dropout의 효과 준다.
# BaggingClassifier 이것도 모델이라서 이 안에 다른 모델을 집어 넣을 수 있다. keras랑 pytorch 도 가능

# Staking
# bagging 방식에서 문제있는 애들을 가중치를 더 올려줘서 다시 훈련시키는 것

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score , r2_score , mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier , VotingClassifier
from sklearn.linear_model import LogisticRegression

# 1.데이터
x, y =load_breast_cancer(return_X_y=True)

x_train , x_test , y_train , y_test = train_test_split(x,y,random_state=777 , train_size=0.8 ,stratify=y )

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2 모델구성
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model = VotingClassifier( estimators= [ ('XGB', xgb) , ('RF',rf) ,('LR',lr) ] ,
                        #  voting='hard' ,# hard가 default
                         voting='soft'
                         )


# 3 훈련
model.fit(x_train ,  y_train )        
# eval_set = [(x_train , y_train) , (x_test, y_test)]  eval_set , verbose ,eval_metric 이 없음

# 4 평가
results = model.score(x_test,y_test)
print('최종점수', results )

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print('add 스코어',acc)

model_class = [ xgb, lr , rf ]

for model2 in model_class:
    model2.fit(x_train,y_train)
    y_pre = model2.predict(x_test)
    score2 = accuracy_score(y_test,y_pre)
    class_name = model2.__class__.__name__
    print('모델이름 : ',class_name , '성능 : ', score2 )
    print("{0} 정확도 : {1:.4f}".format(class_name,score2  ))       
# format(0,1) 을 사용하면 0번째에 있는게 {0} 에 들어가고 1번째에 있는게 score2가 {1.4f}에 들어가고 .4f 로 소수 4번째 자리까지 보여줌

# add 스코어 0.956140350877193

# 최종점수 0.956140350877193
# add 스코어 0.956140350877193

# hard
# 최종점수 0.9824561403508771
# add 스코어 0.9824561403508771

# soft
# 최종점수 0.9824561403508771
# add 스코어 0.9824561403508771

# 최종점수 0.9824561403508771
# add 스코어 0.9824561403508771
# 모델이름 :  XGBClassifier 성능 :  0.9912280701754386
# XGBClassifier 정확도 : 0.9912
# 모델이름 :  LogisticRegression 성능 :  0.9736842105263158
# LogisticRegression 정확도 : 0.9737
# 모델이름 :  RandomForestClassifier 성능 :  0.9736842105263158
# RandomForestClassifier 정확도 : 0.9737