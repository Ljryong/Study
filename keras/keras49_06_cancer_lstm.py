import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score , mean_squared_error , mean_squared_log_error , accuracy_score
from keras.models import Sequential , Model
from keras.layers import Dense , Dropout, Input , Conv2D , Flatten , LSTM
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping , ModelCheckpoint
import matplotlib.pyplot as plt
import datetime
import time

#1 데이터                     ####################################      2진 분류        ###########################################
datasets = load_breast_cancer()
# print(datasets)     
print(datasets.DESCR) # datasets 에 있는 describt 만 뽑아줘
print(datasets.feature_names)       # 컬럼 명들이 나옴

x = datasets.data       # x,y 를 정하는 것은 print로 뽑고 data의 이름과 target 이름을 확인해서 적는 것
y = datasets.target
print(x.shape,y.shape)  # (569, 30) (569,)

x = x.reshape(569,5,6)



es = EarlyStopping(monitor='val_loss' , mode = 'min' , verbose= 1 , patience= 10 ,restore_best_weights=True )

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 0 ,shuffle=True) # 0

print(np.unique(y)) # [0 1]

date = datetime.datetime.now()
date = date.strftime('%m%d-%H%M')
path = 'c:/_data/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path , 'k28_6_', date , '_', filename ])





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
# scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


#2 모델구성
# model = Sequential()
# model.add(Dense(1024 ,input_shape = (30,)))      # sigmoid = 0과 1 사이에서 값이 나온다. 0.5 이상이며 1이 되고 0.5미만이면 0이된다.
# model.add(Dense(512))
# model.add(Dropout(0.5))
# model.add(Dense(256))
# model.add(Dense(128))
# model.add(Dense(1, activation= 'sigmoid'))  # 2진분류 에서는 loss = binary_crossentropy , 
#                                             # activation ='sigmoid'최종 레이어에 써야한다.
#                                             # sigmoid는 중간에도 쓸 수 있고 회귀모델에서도 쓸 수 있다.

#2-1
# input = Input(shape = (30,))
# d1 = Dense(1024)(input)
# d2 = Dense(512)(d1)
# drop1 = Dropout(0.5)(d2)
# d3 = Dense(256)(drop1)
# d4 = Dense(128)(d3)
# output = Dense(1,activation='sigmoid')(d4)
# model = Model(inputs = input , outputs = output)

#2-2
model = Sequential()
model.add(LSTM(150,input_shape = (5,6),activation='relu'))
model.add(Dense(76,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(1))



#3 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy' , 'mse', 'mae'])     
# binary_crossentropy = 2진 분류 = y 가 2개면 무조건 이걸 사용한다
# accuracy = 정확도 = acc
# metrics로 훈련되는걸 볼 수는 있지만 가중치에 영향을 미치진 않는다.

# metrics가 accuracy 
mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose= 1 , save_best_only=True , filepath= filepath)
start_time = time.time()

hist = model.fit(x_train , y_train,epochs = 1000 , batch_size = 100 ,  validation_split= 0.2 , callbacks=[es,mcp] ,)
end_time = time.time()

#4 평가, 예측
loss = model.evaluate(x_test,y_test)        # evaluate = predict로 훈련한 x_test 값을 y_test 값이랑 비교하여 평가한다.
y_predict = model.predict(x_test)
r2 = r2_score(y_test , y_predict)
result = model.predict(x)

print(y_test)
print(np.round(y_predict))                  # round = 반올림 시켜주는 것


def ACC(aaa, bbb) :                                  # aaa,bbb 가 값이 들어가 있는 것이 아니라 '빈 박스' 같은 느낌이다.
    return np.sqrt(mean_squared_error(aaa, bbb))     
acc = ACC(y_test,y_predict)                          # 빈 박스를 여기 ACC() 로 묶어주고 빈 박스의 이름을 정해준 것이다.
print("ACC : ", acc)



def RMSLE(y_test, y_predict) :                                 
    return np.sqrt(mean_squared_log_error(y_test , y_predict))
rmsle = RMSLE(y_test, y_predict)
print("RMSLE : " , rmsle )






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

print("loss = ",loss)
print('r2 = ', r2)
print('시간 :' , end_time - start_time)


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

# Dropout
# ACC :  0.20408860067123857
# RMSLE :  0.14199040067527322
# loss =  [0.30229422450065613, 0.9532163739204407, 0.04165215417742729, 0.0494750551879406]
# r2 =  0.8209948970292394

# cpu
# 시간 : 30.905735731124878
# gpu
# 시간 : 34.68193960189819

# Cnn
# ACC :  36.123033615811764
# RMSLE :  3.059394820581035
# loss =  [5.618140697479248, 0.6315789222717285, 1304.87353515625, 31.911169052124023]
# r2 =  -5606.849455915156
# 시간 : 27.980299472808838


# LSTM
# ACC :  24.153927315699338
# RMSLE :  2.669526662008657
# loss =  [5.618140697479248, 0.6315789222717285, 583.4121704101562, 20.831817626953125]
# r2 =  -2506.2834038419433
# 시간 : 1.6962840557098389



