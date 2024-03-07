# 배깅 보팅 부스팅 스태킹 ? 

# voting 모델을 여러개 사용함
# 2가지의 방식이 있고 soft와 hard가 있음
# hard : 모델을 홀수로 잡아야 된다. 다수결로 결정되는거라 짝수로 잡으면 애매한 상황이 나올 수 있다 // 회귀모델은 hard voting 없다
# soft : 모델을 다 더해서 모델의 수 만큼 나눠서 사용

# bagging
# 모델이 1종류, 하지만 데이터가 다르다
# n_estimators = epoch 랑 다름 // n_estimators 다 다른 데이터 들로 훈련을 함 // n_estimators =  다른데이터로 n_estimators 만큼 훈련 , epoch = 같은데이터로 epoch 만큼 훈련
# n_estimators는 만약 0~9까지 있다면 내가 지정한 수 만큼을 계속 랜덤하게 뽑아내서 훈련함, 중복은 가능하지만 지정한 수가 전부 똑같게는 하지 않음
# 중복이 가능하다는 것은 0,0,0,0,0,0,0... 이런식으로도 가능하다는 뜻
# 빼는 것이 Dropout의 효과 준다.
# BaggingClassifier 이것도 모델이라서 이 안에 다른 모델을 집어 넣을 수 있다. keras랑 pytorch 도 가능

# Staking
# 만약 결측치가 있다면 있는 상태에서 훈련을 시키고 predict을 해서 여러 모델에서 뽑아낸 값을 x로 만들어서 결측치를 채워 훈련할 수 있다.
# 없어도 사용할 수 있고 좋아질 수도 있다. 각 모델에서 예측을 다른 모델의 입력으로 사용하기 때문에 과적합을 방지할 수 있다.

# boosting
# 결과가 좋지 않은 애들만 가중치를 줘서 다시 건든다

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
rn.seed(333)
tf.random.set_seed(333)
np.random.seed(333)
# 1.데이터
x, y =load_breast_cancer(return_X_y=True)

x_train , x_test , y_train , y_test = train_test_split(x,y,random_state=777 , train_size=0.8 ,stratify=y )

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2 모델구성
xgb = XGBClassifier()
rf = RandomForestClassifier()
lr = LogisticRegression()

# models = [xgb,rf,lr]

models = [xgb, rf, lr]  # 모델과 모델 이름을 튜플 형태로 저장

############# 내가 한것 #############
""" y_pred_dict = []

for model, name in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print('모델이름:', name, '성능:', score)
    print("{0} 정확도: {1:.4f}".format(name, score))
    
    y_pred_dict[name] = y_pred

# y_pred_dict = np.array(y_pred_dict).T

# value_list = list(y_pred_dict.values())

print(value_list)

pred = np.concatenate(value_list[0],value_list[1],value_list[2])

print(pred) """
#############  선생님  #############

li = []
li2 = []

for model in models:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_train)          # x_test 는 밑에서 쓰기 위해 train을 넣음
    y_pred_test = model.predict(x_test)
    # print(y_predict.shape)      # (114,)
    li.append(y_predict)
    li2.append(y_pred_test)
    score = accuracy_score(y_test, y_pred_test)  # train으로 predict 시켜서 train 으로 정확도를 확인해야함
    class_name = model.__class__.__name__
    print('모델이름 : ',class_name , '성능 : ', score )
    print("{0} 정확도 : {1:.4f}".format(class_name,score  ))  

# print(li)
new_x_train = np.array(li).T
new_x_test = np.array(li2).T
# print(new_x_train,new_x_train.shape)        # (455, 3)
# print(new_x_test,new_x_test.shape)          # (114, 3)

#####################################

model2 = CatBoostClassifier(verbose=0) 
model2.fit(new_x_train,y_train)
y_pred = model2.predict(new_x_test)
score2 = model2.score(new_x_test,y_test)
class_name = model2.__class__.__name__
print('모델이름 : ',class_name , '성능 : ', score2 )
print("{0} 정확도 : {1:.4f}".format(class_name,score2  ))  

'''============== sklearn의 StackingClassifier =============='''

# model = StackingClassifier([
#     ('xgb',XGBClassifier()),
#     ('RF',RandomForestClassifier()),
#     ('LR',LogisticRegression()),
# ],final_estimator=CatBoostClassifier(verbose=0))

# model.fit(x_train,y_train)
# result = model.score(x_test,y_test)
# print("sklearn Stacking의 ACC : ",result)

# # xgb acc, rf acc, lr acc 가 나와야 됨
# # 스태킹 결과 


# # 모델이름: XGB 성능: 0.9912280701754386
# # XGB 정확도: 0.9912
# # 모델이름: RF 성능: 0.9649122807017544
# # RF 정확도: 0.9649
# # 모델이름: LR 성능: 0.9736842105263158
# # LR 정확도: 0.9737
# # CatBoostClassifier 정확도 : 0.9912
