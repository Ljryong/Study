from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from xgboost import XGBClassifier

class CustomXGBClassifier(XGBClassifier) :              # 상속 XGBClassifier
    def __str__(self):
        return 'XGBClassifier()'

# 해석 CustomXGBClassifier()을 쓰면 XGBClassifier() 라고 나온다

aaa = CustomXGBClassifier()
print(aaa)              # XGBClassifier()
# aaa 는 보통 인스턴스라고 부른다 // 이건 보여주기 위해서 쓴것


#1
x,y = load_iris(return_X_y=True)
print(x.shape,y.shape)


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size= 0.2 , random_state= 2 , shuffle=True, stratify = y )
print(y_test)
print(np.unique(y_test,return_counts=True))

es = EarlyStopping(monitor='val_loss' , mode = 'min' , verbose= 1 , patience=1000000 , restore_best_weights=True)

#2 모델구성
# model = DecisionTreeClassifier(random_state = 777)
# model = RandomForestClassifier(random_state = 777)
# model = GradientBoostingClassifier(random_state = 777)
model= XGBClassifier()

#3 컴파일,훈련
# model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam' , metrics = ['acc'] ) # 값을 뽑으면 metrics에 있는 acc 가 loss 다음으로 나온다.
# model.fit(x_train,y_train,epochs = 10 , batch_size = 1 , verbose = 1 , validation_split=0.2 , callbacks = [es] )
model.fit(x_train,y_train)      # epoch 도 조절할 수 있지만 대부분 default로 둬도 잘됨 바꿔도 큰 차이가 없는게 대부분이다.

#4 평가, 예측
# result = model.evaluate(x_test,y_test)
result = model.score(x_test,y_test)
# evaluate 가 score 로 바뀜 // score 가 보여주는 지표는 accuracy를 보여준다.(분류모델) // 회귀모델에서의 score는 r2 값을 보여준다.
y_predict = model.predict(x_test)


print("model.score" , result)              

print(y_predict)

acc= accuracy_score(y_predict,y_test)
print(acc)

models = [DecisionTreeClassifier(random_state = 777), RandomForestClassifier(random_state = 777) , 
          GradientBoostingClassifier(random_state = 777),CustomXGBClassifier()]
                                                        # CustomXGBClassifier을 여기서 써야 for문에서 print할 때 xgbclassifier로 나옴


############## 훈련 반복 for 문 ###################a
for model in models :
    model.fit(x_train,y_train)
    print('====================')
    result = model.score(x_test,y_test)
    print(model)
    print(model,':',model.feature_importances_ ,result)
   # y_predict = model.predict(x_test)
    print('result',result)

##### 
print(model,':',model.feature_importances_)               # 중요도
# [0.10863948 0.02863402 0.46383013 0.39889637] = 어떤 컬럼이 중요한지를 볼 수 있다.
# DecisionTreeClassifier() : [0.02583333 0.         0.9096407  0.06452597]
# model의 이름을 알 수 있다.

# 1.0
# RandomForestClassifier(random_state=777) : [0.10338254 0.02333183 0.43169059 0.44159505]
# 컬럼을 날릴려할때는 보다 높은 점수의 중요도를 보고 거기서 낮은걸 버리는 것이 좋다.

# 1.0
# GradientBoostingClassifier(random_state=777) : [0.00483338 0.01797268 0.66466787 0.31252608]

# XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=None, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=None, max_leaves=None,
#               min_child_weight=None, missing=nan, monotone_constraints=None,
#               multi_strategy=None, n_estimators=None, n_jobs=None,
#               num_parallel_tree=None, objective='multi:softprob', ...) : [0.01288557 0.01546235 0.905559   0.06609314]


# def ACC(y_test,y_predict) :                 # 순서가 상관이 없다?
#     accuracy_score(y_test,y_predict)
# acc = accuracy_score(y_test,y_predict)

# print("accrucy score : ", acc)

# LinearSVC
# accrucy score :  1.0
# loss =  [0.078158900141716, 0.9333333373069763]

# Perceptron
# model.score 0.5666666666666667

# LogisticRegression
# model.score 1.0

# RandomForestClassifier
# model.score 1.0

# DecisionTreeClassifier
# model.score 0.9333333333333333

# KNeighborsClassifier
# model.score 0.9666666666666667

# models = [LinearSVR(),LinearRegression(),RandomForestRegressor(),DecisionTreeRegressor(),KNeighborsRegressor()]
