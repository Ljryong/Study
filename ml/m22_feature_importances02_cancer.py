import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score , mean_squared_error , mean_squared_log_error , accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron , LogisticRegression , LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#1 데이터                     ####################################      2진 분류        ###########################################
datasets = load_breast_cancer()
# print(datasets)     
print(datasets.DESCR) # datasets 에 있는 describt 만 뽑아줘
print(datasets.feature_names)       # 컬럼 명들이 나옴

x = datasets.data       # x,y 를 정하는 것은 print로 뽑고 data의 이름과 target 이름을 확인해서 적는 것
y = datasets.target
print(x.shape,y.shape)  # (569, 30) (569,)

es = EarlyStopping(monitor='val_loss' , mode = 'min' , verbose= 1 , patience= 100 ,restore_best_weights=True )

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 0 ,shuffle=True) # 0

print(np.unique(y)) # [0 1]




# y 안에 있는 array와 해당 array의 개수들을 알 수 있다.
# numpy 방법
print(np.unique(y, return_counts=True))              # (array([0, 1]), array([212, 357], dtype=int64)) //  0은 212개 1은 357개 


# pandas 방법
print(pd.DataFrame(y).value_counts())               # 3개 다 같지만 행렬일 경우 맨 위에것을 쓰고 아닐경우는 통상적으로 짧은 2번째를 쓴다.
print(pd.value_counts(y))                           
print(pd.Series(y).value_counts())


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2 모델구성
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
models = [DecisionTreeClassifier(random_state = 777), RandomForestClassifier(random_state = 777) , 
          GradientBoostingClassifier(random_state = 777),XGBClassifier()]

############## 훈련 반복 for 문 ###################a
for model in models :
    model.fit(x_train,y_train)
    result = model.score(x_test,y_test)
    print(type(model).__name__,':',model.feature_importances_ ,result)
   # y_predict = model.predict(x_test)
    print(type(model).__name__,'result',result)

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c = 'red' , label = 'loss' , marker='.')
# # c = 'red' , label = 'loss' , marker='.' // c = color , label = 이름 , marker = 1 epoch 당 . 을 찍어주세요
# plt.plot(hist.history['val_loss'], c = 'blue' , label = 'val_loss' , marker='.')

# plt.plot(hist.history['accuracy'], c = 'green' , label = 'accuracy' , marker = '.')

# plt.legend(loc='upper right') # 라벨을 오른쪽 위에 달아주세요
# plt.title('boston loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()

# plt.show()

# print("loss = ",loss)
# print('r2 = ', r2)


# Epoch 21: early stopping
# 6/6 [==============================] - 0s 337us/step - loss: 0.1925 - accuracy: 0.9532 - mse: 0.0526 - mae: 0.1171
# 6/6 [==============================] - 0s 555us/step
# loss =  [0.19248037040233612, 0.9532163739204407, 0.052645422518253326, 0.11708547174930573]
# r2 =  0.773750019420979

# Epoch 100: early stopping
# 6/6 [==============================] - 0s 1ms/step - loss: 0.2468 - accuracy: 0.9123 - mse: 0.0724 - mae: 0.0954
# 6/6 [==============================] - 0s 0s/step
# loss =  [0.2468472272157669, 0.9122806787490845, 0.07242728769779205, 0.09540403634309769]
# r2 =  0.6960611137712087

# Epoch 196: early stopping
# 6/6 [==============================] - 0s 1ms/step - loss: 0.1194 - accuracy: 0.9532 - mse: 0.0359 - mae: 0.0688
# 6/6 [==============================] - 0s 3ms/step
# loss =  [0.11937375366687775, 0.9532163739204407, 0.035903919488191605, 0.0688357949256897]
# r2 =  0.8549507488147026


# MinMaxScaler
# ACC :  0.20191379308357277
# RMSLE :  0.136111044401846
# loss =  [0.5407991409301758, 0.9473684430122375, 0.04076918214559555, 0.04619225487112999]
# r2 =  0.8247895961750011

# StandardScaler
# ACC :  0.21629522817435004
# RMSLE :  0.14992442783510057
# loss =  [12.056479454040527, 0.9532163739204407, 0.046783626079559326, 0.046783626079559326]
# r2 =  0.798941798941799

# MaxAbsScaler
# ACC :  0.1823480652822172
# RMSLE :  0.1261759413013147
# loss =  [0.6252135634422302, 0.9649122953414917, 0.03325081616640091, 0.03857691213488579]
# r2 =  0.8571006558893743

# RobustScaler
# ACC :  0.2243401373378479
# RMSLE :  0.15646226215180234
# loss =  [0.8371858596801758, 0.9473684430122375, 0.05032849311828613, 0.05584089457988739]
# r2 =  0.7837072917060004


# LinearSVC score  0.9415204678362573
# LinearSVC predict  0.9415204678362573

# Perceptron score  0.9532163742690059
# Perceptron predict  0.9532163742690059

# LogisticRegression score  0.9649122807017544
# LogisticRegression predict  0.9649122807017544

# RandomForestClassifier score  0.9590643274853801
# RandomForestClassifier predict  0.9590643274853801

# DecisionTreeClassifier score  0.935672514619883
# DecisionTreeClassifier predict  0.935672514619883

# KNeighborsClassifier score  0.9649122807017544
# KNeighborsClassifier predict  0.9649122807017544



# DecisionTreeClassifier : [0.         0.00919498 0.01016418 0.01584365 0.01609121 0.
#  0.00503366 0.         0.01815419 0.         0.         0.
#  0.         0.04749283 0.00139161 0.         0.         0.
#  0.         0.         0.         0.01067849 0.         0.08615112
#  0.         0.00990228 0.02748451 0.728114   0.0143033  0.        ] 0.9298245614035088
# DecisionTreeClassifier result 0.9298245614035088
# RandomForestClassifier : [0.03891363 0.0140383  0.03879925 0.02668292 0.00514866 0.01635471
#  0.0570186  0.11318238 0.00455979 0.0019195  0.01689234 0.00488786
#  0.00817241 0.02709446 0.00345042 0.00596847 0.00445051 0.00455376
#  0.0050759  0.00704638 0.09266104 0.01481344 0.11453042 0.15156623
#  0.01183975 0.01184728 0.03453495 0.14576949 0.0096946  0.00853255] 0.9649122807017544
# RandomForestClassifier result 0.9649122807017544
# GradientBoostingClassifier : [2.71140099e-04 6.74741353e-03 1.13128776e-04 3.33501310e-04
#  2.19910883e-03 1.16152591e-03 7.53495416e-03 2.58697244e-01
#  7.79296187e-03 2.47512204e-05 2.65212510e-03 1.16735284e-03
#  8.34958923e-04 4.71914901e-02 3.69532631e-03 1.93624706e-03
#  4.45797342e-04 2.18367898e-03 1.80957084e-03 1.99416872e-03
#  2.19634953e-02 2.39722816e-02 7.39748028e-02 3.88736428e-02
#  4.50354610e-03 3.46826962e-03 1.27303211e-02 4.68837572e-01
#  2.53754617e-03 3.52077032e-04] 0.9590643274853801
# GradientBoostingClassifier result 0.9590643274853801
# XGBClassifier : [0.0078288  0.00997445 0.         0.         0.00497615 0.00718123
#  0.04153196 0.25033978 0.00781114 0.00223842 0.01084086 0.
#  0.00738872 0.01997761 0.01375258 0.00541073 0.02006321 0.00065631
#  0.00131278 0.02488152 0.08989877 0.01647127 0.06930902 0.03122866
#  0.00630574 0.00498067 0.03298381 0.304644   0.00352836 0.00448346] 0.9766081871345029
# XGBClassifier result 0.9766081871345029

