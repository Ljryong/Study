# restore_best_weights
# save_best_only
# 에 대한 고찰



from sklearn.datasets import load_boston , load_breast_cancer
import numpy as np
from keras.models import Sequential , load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import pandas as pd

# warning 뜨는것을 없애는 방법, 하지만 아직 왜 뜨는지 모르니 보는것을 추천
import warnings
warnings.filterwarnings('ignore') 

# 현재 사이킷런 버전 1.3.0 보스턴 안됨, 그래서 삭제
# pip uninstall scikit-learn
# pip uninstall scikit-image
# pip uninstall scikit-learn-intelex

# pip install scikit-learn==0.23.2
datasets = load_breast_cancer()

print(datasets)
x = datasets.data
y = datasets.target
print(x)
print(x.shape)      #(506, 13)
print(y)
print(y.shape)      #(506,)

############################
# df = pd.DataFrame(x)
# Nan_num = df.isna().sum()
# print(Nan_num)
############################

print(datasets.feature_names)
# 'CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

print(datasets.DESCR)               
#[실습]
# train , test의 비율을 0.7 이상 0.9 이하
# R2 0.62 이상
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 51, stratify=y )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2


# 3 컴파일, 훈련
from keras.optimizers import Adam
lrs = [1.0,0.1,0.01,0.001,0.0001]

for lr in lrs : 
    model = Sequential()
    model.add(Dense(20,input_dim = 30))
    model.add(Dense(30))
    model.add(Dense(50))
    model.add(Dense(30))
    model.add(Dense(14))
    model.add(Dense(7))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr) )

    from keras.callbacks import EarlyStopping ,ModelCheckpoint , ReduceLROnPlateau
    es = EarlyStopping(monitor='val_loss', mode='min' , patience=10 , verbose= 1 , restore_best_weights= False )

    mcp = ModelCheckpoint(monitor='val_loss' , mode= 'min' , verbose= 1 , save_best_only = True , period = 10 , filepath = '.' ) # mcp 모델만 mcp 에 넣음

    rlr = ReduceLROnPlateau(monitor='val_loss' , mode='auto' , patience= 20 , verbose= 1 , 
                            factor=0.5          # 갱신이 없으면 learning rate를 내가 지정한 수치(0.5) 만큼 나눈다
                                                # learning rate 의 default = 0.001
                                                # 이걸 쓰려면 default 보다 높게 잡고 많이 내려간 뒤 낮아지는게 좋음
                            )

    hist = model.fit(x_train,y_train, epochs = 1000 , batch_size= 32 , validation_split=0.2 , callbacks = [es, mcp, rlr] , verbose = 1 )


    # 4 평가, 예측
    print('============================== 1. 기본 출력 ===============================')

    loss = model.evaluate(x_test,y_test,verbose=0)

    y_predict = model.predict(x_test,verbose=0)
    r2 = r2_score(y_test,y_predict)

    print(loss)
    print('R2 : ' , r2)

    print('==============================')
    print(hist.history['val_loss'])
    print('==============================')


# restore_best_weights
# save_best_only

# True , True 일때  밀린뒤가 없다.  / 체크 포인트들만 저장 
# True , False 일때 멈출 때 까지 전부 저장 
# False , True 일때 체크포인트들만 전부 저장 
# False , False 일때 멈출 때 까지 전부 저장 




# print(hist.history['val_loss'])

# print(end_time - start_time)          # python에서 기본으로 제공하는 시스템
                                        # print는 함수

# #5/5 [==============================] - 0s 0s/step - loss: 23.7223
# 5/5 [==============================] - 0s 4ms/step
# R2 :  0.7506910719128115
# 205.5621416568756
# random = 51 , 20,30,50,30,14,7,1


# minmax
# Epoch 500/500
# 354/354 [==============================] - 0s 511us/step - loss: 22.9468
# 5/5 [==============================] - 0s 0s/step - loss: 25.1686
# 5/5 [==============================] - 0s 0s/step
# R2 :  0.7354911660354422
# 96.14176678657532

# standard
# Epoch 500/500
# 354/354 [==============================] - 0s 614us/step - loss: 22.6931
# 5/5 [==============================] - 0s 748us/step - loss: 23.5351
# 5/5 [==============================] - 0s 0s/step
# R2 :  0.7526578240528379
# 104.04604291915894

# MaxAbsScaler
# Epoch 500/500
# 354/354 [==============================] - 0s 520us/step - loss: 22.1837
# 5/5 [==============================] - 0s 4ms/step - loss: 27.3129
# 5/5 [==============================] - 0s 0s/step
# R2 :  0.7129555609567187
# 101.22957062721252

# RobustScaler
# Epoch 500/500
# 354/354 [==============================] - 0s 529us/step - loss: 22.7813
# 5/5 [==============================] - 0s 499us/step - loss: 26.5415
# 5/5 [==============================] - 0s 784us/step
# R2 :  0.7210621160942314
# 96.14459896087646

