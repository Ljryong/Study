import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error , mean_squared_log_error
import time


#1 데이터(데이터 정제 or 데이터 전처리)

path = "c:/_data/kaggle/bike//"

train_csv = pd.read_csv(path+"train.csv", index_col= 0 )
test_csv = pd.read_csv(path+"test.csv", index_col= 0 )
submission_csv = pd.read_csv(path+"sampleSubmission.csv" )

print(train_csv)        # [10886 rows x 11 columns]
print(test_csv)         # [6493 rows x 8 columns]

print(train_csv.info())

x = train_csv.drop(['count','casual' , 'registered'],axis = 1)
y = train_csv['count']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state=4 , shuffle = True)

#2 모델구성
model = Sequential()
model.add(Dense(1,input_dim = 8))


#3 컴파일, 훈련
model.compile(loss ='mse' , optimizer='adam')
model.fit(x_train , y_train , epochs = 100 , batch_size = 200 , validation_split= 0.2)

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_submit = model.predict(test_csv)

submission_csv['count'] = y_submit

submission_csv.to_csv(path + 'sampleSubmission_0109.csv' , index = False)



# 로스는 :  22717.474609375
# 103/103 [==============================] - 0s 769us/step
# R2 =  0.30513638648798214
# 시간 :  48.2470326423645
# RMSE :  150.72317918064752
# RMSLE :  1.2919153163529318



