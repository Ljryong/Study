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

# aaa = CustomXGBClassifier()
# print(aaa)              # XGBClassifier()
# aaa 는 보통 인스턴스라고 부른다 // 이건 보여주기 위해서 쓴것


#1
datasets = load_iris()
# print(x.shape,y.shape)          # (150, 4) , (150,)
x = datasets.data
y = datasets.target
print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# numpy 컬럼 삭제
# x = np.delete(x,0,axis=1)
# print(x)

# pandas 컬럼 삭제
import pandas as pd
x = pd.DataFrame(x,columns = datasets.feature_names)
x = x.drop('sepal length (cm)', axis=1)
# del x['sepal length (cm)']
# column = x.pop('sepal length (cm)')
print(x)

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

# ====================
# DecisionTreeClassifier(random_state=777)
# DecisionTreeClassifier(random_state=777) : [0.0125     0.01666667 0.4096407  0.56119263] 0.9333333333333333
# result 0.9333333333333333
# ====================
# RandomForestClassifier(random_state=777)
# RandomForestClassifier(random_state=777) : [0.10338254 0.02333183 0.43169059 0.44159505] 1.0
# result 1.0
# ====================
# GradientBoostingClassifier(random_state=777)
# GradientBoostingClassifier(random_state=777) : [0.00483338 0.01797268 0.66466787 0.31252608] 1.0
# result 1.0
# ====================
# XGBClassifier()
# XGBClassifier() : [0.01288557 0.01546235 0.905559   0.06609314] 1.0
# result 1.0
# XGBClassifier() : [0.01288557 0.01546235 0.905559   0.06609314]


# ====================
# DecisionTreeClassifier(random_state=777)
# DecisionTreeClassifier(random_state=777) : [0.02916667 0.4096407  0.56119263] 0.9333333333333333
# result 0.9333333333333333
# ====================
# RandomForestClassifier(random_state=777)
# RandomForestClassifier(random_state=777) : [0.12710911 0.49330376 0.37958713] 0.9666666666666667
# result 0.9666666666666667
# ====================
# GradientBoostingClassifier(random_state=777)
# GradientBoostingClassifier(random_state=777) : [0.01930045 0.64036571 0.34033384] 1.0
# result 1.0
# ====================
# XGBClassifier()
# XGBClassifier() : [0.01639269 0.91807455 0.06553278] 1.0
# result 1.0
# XGBClassifier() : [0.01639269 0.91807455 0.06553278]