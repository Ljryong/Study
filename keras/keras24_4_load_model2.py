# 9_1 복붙

from sklearn.datasets import load_boston
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
datasets = load_boston()


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
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 51 )

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
# model = Sequential()
# model.add(Dense(20,input_dim = 13))
# model.add(Dense(30))
# model.add(Dense(50))
# model.add(Dense(30))
# model.add(Dense(14))
# model.add(Dense(7))
# model.add(Dense(1))
# model.save("../_data/_save/keras24_save_model.h5")

model = load_model('../_data/_save/keras24_3_save_model2.h5')

# model = load_model('../_data/_save/keras24_save_model.h5')

model.summary()

# model.save("c:/_data/_save/keras24_save_model.h5")

# model.save("c:/_data/keras24_save_model.h5")    # 절대 경로

# model.save("./_data/_save/keras24_save_model.h5")     # 현재 작업하고 있는 폴더(Study)에 생성 , 상대 경로

# model.save("../_data/_save/keras24_save_model.h5")      # _data 상위 폴더에 생성 (c 드라이브) , 상대 경로





# #3
# model.compile(loss='mse', optimizer='adam')
# start_time = time.time()
# model.fit(x_train,y_train,epochs=100,batch_size=1)
# end_time = time.time()


# model.save("../_data/_save/keras24_3_save_model2.h5")


#4
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print(loss)
print('R2 : ' , r2)
# print(end_time - start_time)            # python에서 기본으로 제공하는 시스템
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










