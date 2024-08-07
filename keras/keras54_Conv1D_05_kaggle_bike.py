from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential , Model
from keras.layers import Dense , Dropout , Input ,Conv2D , Flatten , LSTM ,Conv1D
from keras.callbacks import EarlyStopping , ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time

#1 데이터

path = 'c:/_data/kaggle/bike//'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')


print(train_csv.shape)      # (10886, 11)

print(test_csv.shape)       # (6493, 8)

print(train_csv.isnull().sum()) 
print(test_csv.isna().sum())

x = train_csv.drop(['casual' , 'registered', 'count'], axis= 1 )        # [6493 rows x 8 columns] // drop을 줄 때 '를 따로 따로 줘야된다.
y = train_csv['count']

print(x)
print(y)            #  10886, 

date = datetime.datetime.now()
date = date.strftime('%m%d-%H%M')
path = 'c:/_data/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path , 'k28_5_', date , '_', filename ])

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state= 6974 ) #7
# x_train_d, x_val , y_train_d, y_val  = train_test_split(x_train, y_train, train_size=0.8, random_state=10)

es = EarlyStopping(monitor = 'val_loss' , mode = 'min', patience = 10 , verbose= 1 ,restore_best_weights=True )
mcp = ModelCheckpoint(monitor = 'val_loss' , mode = 'min' , verbose = 1 , save_best_only=True , filepath= filepath )
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

print(x_train.shape,x_test.shape)           # (7620, 8) (3266, 8)
print(test_csv)                             # (6493, 8)


print(x_train.shape,x_test.shape)  
print(test_csv.shape)
x_train = x_train.values.reshape(7620,4,2)
x_test = x_test.values.reshape(3266,4,2)
test_csv = test_csv.values.reshape(6493,4,2)

###################
# # scaler = MinMaxScaler()
# scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)






#2 모델구성
# model = Sequential()            # relu 0이하는 전부 0으로 바꾸고 양수는 그대로 놔둔다. 
# model.add(Dense(512, input_shape = (8,) ))
# model.add(Dense(256, activation= 'relu'))
# model.add(Dropout(0.4))
# model.add(Dense(128))
# model.add(Dropout(0.4))
# model.add(Dense(64))
# model.add(Dropout(0.4))
# model.add(Dense(32,activation= 'relu'))
# model.add(Dense(16,activation= 'relu'))
# model.add(Dense(1))
# default 값으로 linear(선형의)가 존재한다.
# 마지막에는 relu를 잘 쓰지 않는다. 최종 아웃풋에 자주 쓰는 애는 'softmax' 라고 따로 존재한다
# 마지막에도 relu를 쓰면 오류가 덜 뜨긴 하지만 성능이 안좋아짐, 반대로 안쓰면 성능이 조금 좋아지지만 오류가 더 많이 뜬다.

#2-1
# input = Input(shape=(8,))
# d1 = Dense(512)(input)
# d2 = Dense(256,activation='relu')(d1)
# drop1 = Dropout(0.4)(d2)
# d3 = Dense(128)(drop1)
# drop2 = Dropout(0.4)(d3)
# d4 = Dense(64)(drop2)
# drop3 = Dropout(0.4)(d4)
# d5 = Dense(32,activation='relu')(drop3)
# d6 = Dense(16,activation='relu')(d5)
# output = Dense(1)(d6)
# model = Model(inputs = input , outputs = output)

# 2-2
model = Sequential()
model.add(Conv1D(20,2,input_shape = (4,2)))
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer='adam', metrics = ['mse' , 'mae'])


start_time = time.time()

hist = model.fit(x_train, y_train, epochs = 1000 , batch_size= 100, verbose= 1 , validation_split=0.2 , callbacks = [es,mcp])
end_time = time.time()

#4 평가, 예측
loss = model.evaluate(x_test,y_test)

y_submit = model.predict(test_csv)

print(y_submit)
print(y_submit.shape)       # (6493, 1)

# 결과 넣기

submission_csv['count'] = y_submit


print(submission_csv)             # [6493 rows x 2 columns]

submission_csv.to_csv(path + "sampleSubmission_0110.csv" , index = False)



print("로스는 : " , loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 = " ,r2)
print('시간 : ',end_time - start_time)

################### 데이터 프레임 조건 중요 ###################
print("음수갯수",submission_csv[submission_csv['count']<0].count())    





# plt.figure(figsize = (9,6))
# plt.plot(hist.history['loss'], c = 'red' , label = 'loss' , marker = '.')
# plt.plot(hist.history['val_loss'],c = 'blue' , label = 'val_loss' , marker = '.')
# plt.legend(loc = 'upper right')


# print(hist)
# plt.title('kaggle loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()

# plt.show()


# [6493 rows x 2 columns]
# 로스는 :  [22091.794921875, 22091.794921875, 107.92926025390625]
# 103/103 [==============================] - 0s 503us/step
# R2 =  0.29941292536061126



# # MinMaxScaler
# [6493 rows x 2 columns]
# 로스는 :  [22061.03515625, 22061.03515625, 109.0513916015625]
# 103/103 [==============================] - 0s 717us/step
# R2 =  0.3003884960999149

# # StandardScaler
# [6493 rows x 2 columns]
# 로스는 :  [21999.53515625, 21999.53515625, 109.65789794921875]
# 103/103 [==============================] - 0s 786us/step
# R2 =  0.3023387950309775

# # MaxAbsScaler
# [6493 rows x 2 columns]
# 로스는 :  [21497.927734375, 21497.927734375, 109.61567687988281]
# 103/103 [==============================] - 0s 517us/step
# R2 =  0.31824596802694793

# # RobustScaler
# [6493 rows x 2 columns]
# 로스는 :  [21784.3046875, 21784.3046875, 108.29178619384766]
# 103/103 [==============================] - 0s 582us/step
# R2 =  0.3091640872462358



# Dropout
# [6493 rows x 2 columns]
# 로스는 :  [23057.0, 23057.0, 112.02766418457031]
# 103/103 [==============================] - 0s 502us/step
# R2 =  0.26880372940966746


# cpu
# 시간 :  148.1658980846405
# gpu
# 시간 :  142.24108934402466


# Cnn
# 로스는 :  [23889.271484375, 23889.271484375, 116.04253387451172]
# 103/103 [==============================] - 0s 648us/step
# R2 =  0.24241048760682415
# 시간 :  8.87637734413147
# 음수갯수 datetime    182
# count       182


# LSTM
# [6493 rows x 2 columns]
# 로스는 :  [21968.3203125, 21968.3203125, 110.08675384521484]
# 103/103 [==============================] - 0s 773us/step
# R2 =  0.3033285494226523
# 시간 :  19.984963178634644
# 음수갯수 datetime    0
# count       0
# dtype: int64

# Conv1D
# [6493 rows x 2 columns]
# 로스는 :  [23865.8671875, 23865.8671875, 116.0575180053711]
# 103/103 [==============================] - 0s 474us/step
# R2 =  0.24315265498677285
# 시간 :  9.174778461456299
# 음수갯수 datetime    143
# count       143
# dtype: int64