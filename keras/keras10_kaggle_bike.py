# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error , mean_squared_log_error
import time


#1 데이터(데이터 정제 or 데이터 전처리)

path = "C:/_data/kaggle/bike//"

train_csv = pd.read_csv(path + "train.csv" , index_col = 0 )
print(train_csv)

test_csv = pd.read_csv(path + "test.csv", index_col = 0 )
print(test_csv)

submission_csv = pd.read_csv(path + "sampleSubmission.csv" )


print(train_csv.shape)      # (10886, 11)

print(test_csv.shape)       # (6493, 8)

print(train_csv.isnull().sum()) 
print(test_csv.isna().sum())

x = train_csv.drop(['casual' , 'registered', 'count'], axis= 1 )        # [6493 rows x 8 columns] // drop을 줄 때 '를 따로 따로 줘야된다.
y = train_csv['count']

print(x)
print(y)            #  10886, 

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state= 13256 ) #7
# x_train_d, x_val , y_train_d, y_val  = train_test_split(x_train, y_train, train_size=0.8, random_state=10)



#2 모델구성
model = Sequential()
model.add(Dense(20,input_dim = 8 , activation='relu'))                  # relu 0이하는 전부 0으로 바꾸고 양수는 그대로 놔둔다. 
model.add(Dense(30, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))                                                     # default 값으로 linear(선형의)가 존재한다.
# 마지막에는 relu를 잘 쓰지 않는다. 최종 아웃풋에 자주 쓰는 애는 'softmax' 라고 따로 존재한다
# 마지막에도 relu를 쓰면 오류가 덜 뜨긴 하지만 성능이 안좋아짐, 반대로 안쓰면 성능이 조금 좋아지지만 오류가 더 많이 뜬다.

#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer='adam')

start_time = time.time()
model.fit(x_train, y_train, epochs = 300 , batch_size= 50, verbose= 1 )
end_time = time.time()

#4 평가, 예측
loss = model.evaluate(x_test,y_test)

y_submit = model.predict(test_csv)

print(y_submit)
print(y_submit.shape)       # (6493, 1)

# 결과 넣기

submission_csv['count'] = y_submit


print(submission_csv)             # [6493 rows x 2 columns]

submission_csv.to_csv(path + "sampleSubmission_0108.csv" , index = False)



print("로스는 : " , loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 = " ,r2)
print("시간 : " , end_time - start_time)

################### 데이터 프레임 조건 중요 ###################
print("음수갯수",submission_csv[submission_csv['count']<0].count())    # sampleSubmission_csv 중 'count'에서 0 보다 작은 데이터 값을 세라



def RMSE(y_test, y_predict) :                                          # python에선 함수를 만들때 앞에 def 를 쓴다 // 이 함수는 지금 결과값에 영향을 주지 않는다.
    return np.sqrt(mean_squared_error(y_test , y_predict))             # MSE 결과값에 루트를 씌우준다.
rmse = RMSE(y_test,y_predict)    
print("RMSE : ", rmse)

def RMSLE(y_test, y_predict) :                                 
    return np.sqrt(mean_squared_log_error(y_test , y_predict))
rmsle = RMSLE(y_test, y_predict)
print("RMSLE : " , rmsle )



# y_submit=abs(y_submit)              # y_submit에 절대값 씌워주기


""" y_submit 의 음수 값을 0 으로 바꾸는 방법 = 함수[함수 < 0 ] = 0 
y_submit 의 음수 값을 결측치로 바꾸는 방법 = 함수[함수 < 0 ] = np.NaN


y_submit[y_submit < 0] = 0
 """
 
# 여기서 하는 처리는 후처리 라고 하고 후처리는 데이터 조작으로 하지 않는다. 음수가 나오거나 나올거 같으면 훈련 과정에서 처리를 해줘야한다.
# Relu = 활성화 함수 -- 후처리가 아닌 모델구성에서 역전파로 올라갈때 ReLU를 사용한다.



# 로스는 :  24193.734375
# 103/103 [==============================] - 0s 342us/step
# R2 =  0.2655433866648905
# epochs = 400 , batch_size= 50
# model.add(Dense(10,input_dim = 8))
# model.add(Dense(25))
# model.add(Dense(50))
# model.add(Dense(25))
# model.add(Dense(15))
# model.add(Dense(7))
# model.add(Dense(4))
# model.add(Dense(1))
# random_state= 69 


# [6493 rows x 2 columns]
# 로스는 :  23641.580078125
# 103/103 [==============================] - 0s 480us/step
# R2 =  0.32094848078170934
# 음수갯수 datetime    0
# count       0
# dtype: int64


# [6493 rows x 2 columns]
# 로스는 :  21660.03515625
# 103/103 [==============================] - 0s 731us/step
# R2 =  0.3372962092161955
# 음수갯수 datetime    0
# count       0
# dtype: int64
# RMSE :  147.17349400130269

