# restore_best_weights
# save_best_only
# 에 대한 고찰



from sklearn.datasets import load_boston
import numpy as np
from keras.models import Sequential , load_model
from keras.layers import Dense , Dropout , Conv2D , Flatten
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
datasets = load_boston()

# print(datasets)
x = datasets.data
y = datasets.target
# print(x)
# print(x.shape)      #(506, 13)
# print(y)
# print(y.shape)      #(506,)


x = x.reshape(506,13,1,1)
# y = y.reshape(506,1,1,1)

print(x.shape)


############################
# df = pd.DataFrame(x)
# Nan_num = df.isna().sum()
# print(Nan_num)
############################

# print(datasets.feature_names)
# 'CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

# print(datasets.DESCR)               
#[실습]
# train , test의 비율을 0.7 이상 0.9 이하
# R2 0.62 이상
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 51 )


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2
model = Sequential()
model.add(Conv2D(20,(2,1),input_shape = (13,1,1) , activation='relu' , padding='valid' , strides=1  ))
#####################################################################################################################################
                  # 커널사이즈를 맞춰주기 (13,1,1) 에는 (2,2) 가 들어갈 수가 없다. 중요
#####################################################################################################################################
model.add(Dropout(0.2))             # dropout에 퍼센트를 사용한다 0.2 = 20퍼 // Dropout 위에 있는 레이어에서 정해준 퍼센트만큼을 뺀다. 
                                    # evaluate에는 dropout이 들어가지 않는다. 훈련에서는 적용되지만 평가와 예측에서는 적용되지 않는다.
model.add(Conv2D(102,(2,2),padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(16,(2,2),padding='same'))
model.add(Flatten())
model.add(Dense(74))
model.add(Dropout(0.2))
model.add(Dense(7))
model.add(Dense(1))
# model.summary()

# 3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping ,ModelCheckpoint
import datetime
date= datetime.datetime.now()       # <class 'datetime.datetime'>
print(date)                         # 2024-01-17 10:52:55.510197
date = date.strftime('%m%d_%H%M')       # m = 달 , d = day 날 , H = 시간 , M = 분
print(date)                         # <calss'str'>
path ='../_data/_save/MCP/'     # 문자를 저장한것이다
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'        # d = 정수 , f = 소수점  04d = 4자리숫자까지 ,04f = 소수 4자리까지 / ex) 1000-0.3333.hdf5

filepath = ''.join([path ,'k28_01_', date , '_' , filename])        # join = 이어주는것 
# = '../_data/_save/MCP/k25_0117_1058_0101-0.3333.hdf5'


es = EarlyStopping(monitor='val_loss', mode='min' , patience=10 , verbose= 1 , restore_best_weights= False )

mcp = ModelCheckpoint(monitor='val_loss' , mode= 'min' , verbose= 1 , save_best_only = True , filepath = filepath ) # mcp 모델만 mcp 에 넣음
start_time = time.time()
hist = model.fit(x_train,y_train, epochs = 1000 , batch_size= 1000 , validation_split=0.2 , callbacks = [ es , mcp] , verbose = 1 )
end_time = time.time()

# # 체크되는 지점마다 저장 // mode를 모르겠으면 auto 로 주면 된다. // save_best_only 가장 좋은 애만 저장한다.

# model.save('../_data/_save/keras25_3_save_model.h5')    # 그냥 save 는 h5 , mcp 는 hdf5

# model = load_model('../_data/_save/MCP/keras25_MCP1.hdf5')



# 4 평가, 예측
print('============================== 1. 기본 출력 ===============================')

loss = model.evaluate(x_test,y_test,verbose=0)

y_predict = model.predict(x_test,verbose=0)
r2 = r2_score(y_test,y_predict)
print('시간 : ',end_time - start_time)
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

# Dropout
# ============================== 1. 기본 출력 ===============================
# 30.629240036010742
# R2 :  0.6781025367432623
# ==============================
# [358.94464111328125, 320.5968017578125, 292.8230285644531, 263.2049255371094, 199.89134216308594, 120.7367935180664, 83.13162231445312, 234.57949829101562, 59.34804153442383, 47.6052131652832, 77.88818359375, 35.67835998535156, 38.44282150268555, 44.69051742553711, 36.69685363769531, 31.060993194580078, 31.68844223022461, 33.02342224121094, 29.49051284790039, 29.08911895751953, 28.633325576782227, 27.367809295654297, 31.146072387695312, 26.997100830078125, 29.934904098510742, 36.503360748291016, 40.764739990234375, 47.59799575805664, 30.998315811157227, 32.49274444580078, 37.87171936035156, 29.916290283203125, 28.604448318481445, 26.374605178833008, 26.755233764648438, 29.074689865112305, 28.55221939086914, 26.318973541259766, 26.506406784057617, 26.30133056640625, 26.428823471069336, 37.75775909423828, 41.37565994262695, 26.30621337890625, 25.411434173583984, 25.91392707824707, 27.403852462768555, 25.90078353881836, 31.636533737182617, 33.431556701660156, 27.90660858154297, 29.76919174194336, 28.134960174560547, 26.96661376953125, 30.082021713256836]
# ==============================


# cpu
# 시간 :  20.883222818374634
# gpu
# 시간 :  19.5326406955719



# Cnn
# ============================== 1. 기본 출력 ===============================
# 시간 :  23.250208854675293
# 48.254432678222656
# R2 :  0.49287093490299305
# ==========================================================================