from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold , StratifiedKFold

#1
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
y = y 

print(x.shape,y.shape)      # (581012, 54) (581012,)
print(pd.value_counts(y))   # 2    283301 , 1    211840 , 3     35754 , 7     20510 , 6     17367 , 5      9493 , 4      2747   (n,7)

x_train , x_test , y_train , y_test = train_test_split(x,y ,test_size=0.3 , random_state= 2222 ,shuffle=True, stratify=y ) # 0

es= EarlyStopping(monitor='val_loss' , mode = 'min', verbose= 1 ,patience=10, restore_best_weights=True )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

'''parameters  = {'n_estimater' : [400,500,1000], # 디폴트 100 / 1~inf / 정수
'learning_rate' : [0.1,0.2,0.01,0.001], # 디폴트 0.3 / 0~1 / eta 제일 중요  
# learning_rate(훈련율) : 작을수록 디테일하게 보고 크면 클수록 듬성듬성 본다. batch_size랑 비슷한 느낌
#                        하지만 너무 작으면 오래 걸림 데이터의 따라 잘 조절 해야된다
'max_depth' : [2,3,4,5], # 디폴트 6 / 0~inf / 정수    # tree의 깊이를 나타냄
'gamma' : [1,2,3,4,5,7,10,100], # 디폴트 0 / 0~inf 
'min_child_weight' : [0,0.5,1,5,10,100], # 디폴트 1 / 0~inf 
'subsample' : [0,0.1,0.2,0.3,0.5,0.7,1], # 디폴트 1 / 0~1
'colsample_bytree' : [0.1,0.2,0.3,0.5,0.7,1], # 디폴트 1 / 0~1 
'colsample_bylevel' : [0,0.1,0.2,0.3,0.5,0.7,1], # 디폴트 1 / 0~1
'colsample_bynode' : [0,0.1,0.2,0.3,0.5,0.7,1], # 디폴트 1 / 0~1
'reg_alpha' : [0,0.1,0.01,0.001,1,2,10], # 디폴트 0 / 0~inf / L1 절대값 가중치 규제 / alpha / 중요
'reg_lambda' : [0,0.1,0.01,0.001,1,2,10],# 디폴트 1 / 0~inf / L2 제곱 가중치 규제 / lambda / 중요
}'''

n_splits=5
kfold = StratifiedKFold(n_splits= n_splits , random_state=2222, shuffle=True )

#2
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier , XGBRegressor

# model = RandomizedSearchCV ,  , cv = kfold ,random_state=2222 )
model = XGBRegressor(tree_method='gpu_hist' , random_state = 777)

#3 훈련
model.fit(x_train,y_train-1)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
# print('매겨변수' , model.best_estimator_)
# print('매겨변수' , model.best_params_)

# 초기 특성 중요도
import warnings
warnings.filterwarnings('ignore')
feature_importances = model.feature_importances_
sort= np.argsort(feature_importances)               # argsort 열의 번호로 반환해줌
print(sort)

removed_features = 0

# 각 반복에서 피처를 추가로 제거하면서 성능 평가
for i in range(len(model.feature_importances_) - 1):
    remove = sort[:i+1]  # 추가로 제거할 피처의 인덱스
    
    print(f"Removing features at indices: {remove}")
    
    # 해당 특성 제거
    x_train_removed = np.delete(x_train, remove, axis=1)
    x_test_removed = np.delete(x_test, remove, axis=1)

    # 모델 재구성 및 훈련
    model.fit(x_train_removed, y_train-1, eval_set=[(x_train_removed, y_train-1), (x_test_removed, y_test-1)],
              verbose=0, eval_metric='mlogloss', early_stopping_rounds=10)
    
    # 모델 평가
    acc = model.score(x_test_removed, y_test)
    print('Accuracy after removing features:', acc)
    
    # 제거된 피처의 개수를 누적
    removed_features += 1
    print(f"Total number of removed features: {removed_features}\n")




# model.score 0.10956145584725537



#  warnings.warn(smsg, UserWarning)
# model.score 0.0579447402239765
# [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34 44 13 36 42 11 12 52 51 17
#  45 15 25 35 10  0]
# Removing features at indices: [28]
# Accuracy after removing features: 0.0579447402239765
# Total number of removed features: 1

# Removing features at indices: [28 20]
# Accuracy after removing features: 0.05854139893519369
# Total number of removed features: 2

# Removing features at indices: [28 20 27]
# Accuracy after removing features: 0.05817422434367542
# Total number of removed features: 3

# Removing features at indices: [28 20 27 14]
# Accuracy after removing features: 0.05825454378557004
# Total number of removed features: 4

# Removing features at indices: [28 20 27 14 41]
# Accuracy after removing features: 0.05841518266935928
# Total number of removed features: 5

# Removing features at indices: [28 20 27 14 41 49]
# Accuracy after removing features: 0.056957958509271156
# Total number of removed features: 6

# Removing features at indices: [28 20 27 14 41 49  2]
# Accuracy after removing features: 0.057497246190563615
# Total number of removed features: 7

# Removing features at indices: [28 20 27 14 41 49  2 19]
# Accuracy after removing features: 0.05795047732696897
# Total number of removed features: 8

# Removing features at indices: [28 20 27 14 41 49  2 19 18]
# Accuracy after removing features: 0.05702106664218836
# Total number of removed features: 9

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8]
# Accuracy after removing features: 0.06012483936111621
# Total number of removed features: 10

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31]
# Accuracy after removing features: 0.0574570864696163
# Total number of removed features: 11

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22]
# Accuracy after removing features: 0.05793900312098403
# Total number of removed features: 12

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47]
# Accuracy after removing features: 0.05810537910776574
# Total number of removed features: 13

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32]
# Accuracy after removing features: 0.05664815494767762
# Total number of removed features: 14

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50]
# Accuracy after removing features: 0.05917821736735818
# Total number of removed features: 15

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1]
# Accuracy after removing features: 0.057692307692307696
# Total number of removed features: 16

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4]
# Accuracy after removing features: 0.059074949513493666
# Total number of removed features: 17

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39]
# Accuracy after removing features: 0.05761772535340554
# Total number of removed features: 18

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6]
# Accuracy after removing features: 0.062241830365338716
# Total number of removed features: 19

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29]
# Accuracy after removing features: 0.060916559574077475
# Total number of removed features: 20

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24]
# Accuracy after removing features: 0.06128373416559574
# Total number of removed features: 21

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30]
# Accuracy after removing features: 0.06022237011198825
# Total number of removed features: 22

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21]
# Accuracy after removing features: 0.06215577382045163
# Total number of removed features: 23

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53]
# Accuracy after removing features: 0.06195497521571507
# Total number of removed features: 24

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26]
# Accuracy after removing features: 0.06037153478979255
# Total number of removed features: 25

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7]
# Accuracy after removing features: 0.06336056544887093
# Total number of removed features: 26

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9]
# Accuracy after removing features: 0.07787543601982742
# Total number of removed features: 27

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38]
# Accuracy after removing features: 0.0784319350100973
# Total number of removed features: 28

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3]
# Accuracy after removing features: 0.08848907655590234
# Total number of removed features: 29

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5]
# Accuracy after removing features: 0.10026734899944924
# Total number of removed features: 30

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40]
# Accuracy after removing features: 0.1018221039104094
# Total number of removed features: 31

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33]
# Accuracy after removing features: 0.10455870203781899
# Total number of removed features: 32

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23]
# Accuracy after removing features: 0.10575775656324582
# Total number of removed features: 33

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16]
# Accuracy after removing features: 0.10477097484854048
# Total number of removed features: 34

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37]
# Accuracy after removing features: 0.10615935377271893
# Total number of removed features: 35

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48]
# Accuracy after removing features: 0.10662405911510923
# Total number of removed features: 36

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46]
# Accuracy after removing features: 0.10460459886175877
# Total number of removed features: 37

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43]
# Accuracy after removing features: 0.10519552046998348
# Total number of removed features: 38

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34]
# Accuracy after removing features: 0.10354323480815128
# Total number of removed features: 39

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34 44]
# Accuracy after removing features: 0.10422021296126308
# Total number of removed features: 40

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34 44 13]
# Accuracy after removing features: 0.10418579034330824
# Total number of removed features: 41

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34 44 13 36]
# Accuracy after removing features: 0.11256769781531117
# Total number of removed features: 42

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34 44 13 36 42]
# Accuracy after removing features: 0.111121947861208
# Total number of removed features: 43

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34 44 13 36 42 11]
# Accuracy after removing features: 0.11107031393427574
# Total number of removed features: 44

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34 44 13 36 42 11 12]
# Accuracy after removing features: 0.10993436754176611
# Total number of removed features: 45

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34 44 13 36 42 11 12 52]
# Accuracy after removing features: 0.11042775839911878
# Total number of removed features: 46

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34 44 13 36 42 11 12 52 51]
# Accuracy after removing features: 0.11060560859188544
# Total number of removed features: 47

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34 44 13 36 42 11 12 52 51 17]
# Accuracy after removing features: 0.11290618689186709
# Total number of removed features: 48

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34 44 13 36 42 11 12 52 51 17
#  45]
# Accuracy after removing features: 0.11528708463374335
# Total number of removed features: 49

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34 44 13 36 42 11 12 52 51 17
#  45 15]
# Accuracy after removing features: 0.11700247842849275
# Total number of removed features: 50

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34 44 13 36 42 11 12 52 51 17
#  45 15 25]
# Accuracy after removing features: 0.11688773636864329
# Total number of removed features: 51

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34 44 13 36 42 11 12 52 51 17
#  45 15 25 35]
# Accuracy after removing features: 0.12240109234440977
# Total number of removed features: 52

# Removing features at indices: [28 20 27 14 41 49  2 19 18  8 31 22 47 32 50  1  4 39  6 29 24 30 21 53
#  26  7  9 38  3  5 40 33 23 16 37 48 46 43 34 44 13 36 42 11 12 52 51 17
#  45 15 25 35 10]
# Accuracy after removing features: 0.10921149256471452
# Total number of removed features: 53
