from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense , Dropout
from keras.callbacks import EarlyStopping , ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


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
mcp = ModelCheckpoint(monitor = 'val_loss' , mode = 'min' , verbose = 0 , save_best_only=True , filepath= filepath )
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# # scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2 모델구성

# default 값으로 linear(선형의)가 존재한다.
# 마지막에는 relu를 잘 쓰지 않는다. 최종 아웃풋에 자주 쓰는 애는 'softmax' 라고 따로 존재한다
# 마지막에도 relu를 쓰면 오류가 덜 뜨긴 하지만 성능이 안좋아짐, 반대로 안쓰면 성능이 조금 좋아지지만 오류가 더 많이 뜬다.

#3 컴파일, 훈련
from keras.optimizers import Adam
learning_rates = [ 1.0, 0.1, 0.01, 0.001, 0.0001 ]

rlr = ReduceLROnPlateau(monitor='val_loss' , mode='auto' , patience= 20 , verbose= 1 , 
                        factor=0.5          # 갱신이 없으면 learning rate를 내가 지정한 수치(0.5) 만큼 나눈다
                                            # learning rate 의 default = 0.001
                                            # 이걸 쓰려면 default 보다 높게 잡고 많이 내려간 뒤 낮아지는게 좋음
                        )

for learning_rate in learning_rates :
    model = Sequential()            # relu 0이하는 전부 0으로 바꾸고 양수는 그대로 놔둔다. 
    model.add(Dense(512, input_dim = 8 ))
    model.add(Dense(256, activation= 'swish'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation= 'swish'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation= 'swish'))
    model.add(Dropout(0.4))
    model.add(Dense(32,activation= 'swish'))
    model.add(Dense(16,activation= 'swish'))
    model.add(Dense(1)) 
    
    model.compile(loss = 'mse' , optimizer=Adam(learning_rate) , metrics = ['mse' , 'mae'])
    hist = model.fit(x_train, y_train, epochs = 10000 , batch_size= 10, verbose= 0 , validation_split=0.2 , callbacks = [es,mcp,rlr])

    #4 평가, 예측
    loss = model.evaluate(x_test,y_test)
    y_predict = model.predict(x_test)
    
    print('lr : {0}, 로스 : {1} '.format(learning_rate,loss[0]))
    r2 = r2_score(y_test,y_predict)
    print('lr : {0} , R2 : {1} '.format(learning_rate,r2))






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







