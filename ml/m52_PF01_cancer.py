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
# 결측치를 빼두고 그걸 y로 두고 훈련을 먼저 시킨다
# 그 후 나온 값으로 훈련을 시킨다 

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler ,StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score , r2_score , mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier , VotingClassifier , StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import random as rn
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
rn.seed(333)
tf.random.set_seed(333)
np.random.seed(333)

# 1.데이터
x, y =load_breast_cancer(return_X_y=True)

pf = PolynomialFeatures( degree = 2 , include_bias=False )
x_poly = pf.fit_transform(x)

x_train , x_test , y_train , y_test = train_test_split(x_poly,y,random_state=777 , train_size=0.8 ,stratify=y )

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2 모델구성
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

model = RandomForestClassifier()

#3 훈련
model.fit(x_train,y_train)

#4 평가, 예측
y_pred = model.predict(x_test)
print('model.score : ' , model.score(x_test,y_test) )
print('accuracy : ' , accuracy_score(y_test,y_pred) )

# model.score :  0.9824561403508771
# accuracy :  0.9824561403508771

# PolynomialFeatures 
# model.score :  0.9912280701754386
# accuracy :  0.9912280701754386