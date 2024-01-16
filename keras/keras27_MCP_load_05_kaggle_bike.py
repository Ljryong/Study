from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential , load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping , ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state= 6974 ) #7
# x_train_d, x_val , y_train_d, y_val  = train_test_split(x_train, y_train, train_size=0.8, random_state=10)

es = EarlyStopping(monitor = 'val_loss' , mode = 'min', patience = 10 , verbose= 1 ,restore_best_weights=True )
mcp = ModelCheckpoint(monitor = 'val_loss' , mode = 'min' , verbose = 1 , save_best_only=True , filepath='c:/_data/_save/MCP/keras26_MCP5.hdf5')
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
# model = Sequential()            # relu 0이하는 전부 0으로 바꾸고 양수는 그대로 놔둔다. 
# model.add(Dense(512, input_dim = 8 ))
# model.add(Dense(256, activation= 'relu'))
# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(32,activation= 'relu'))
# model.add(Dense(16,activation= 'relu'))
# model.add(Dense(1))
# # default 값으로 linear(선형의)가 존재한다.
# # 마지막에는 relu를 잘 쓰지 않는다. 최종 아웃풋에 자주 쓰는 애는 'softmax' 라고 따로 존재한다
# # 마지막에도 relu를 쓰면 오류가 덜 뜨긴 하지만 성능이 안좋아짐, 반대로 안쓰면 성능이 조금 좋아지지만 오류가 더 많이 뜬다.

# #3 컴파일, 훈련
# model.compile(loss = 'mse' , optimizer='adam', metrics = ['mse' , 'mae'])




# hist = model.fit(x_train, y_train, epochs = 10000 , batch_size= 10, verbose= 1 , validation_split=0.2 , callbacks = [es])


model = load_model('c:/_data/_save/MCP/keras26_MCP5.hdf5')
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












