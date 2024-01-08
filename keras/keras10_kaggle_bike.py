# https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1 데이터

path = "C:/_data/kaggle/bike//"

train_csv = pd.read_csv(path + "train.csv" , index_col = 0 )
print(train_csv)

test_csv = pd.read_csv(path + "test.csv", index_col = 0 )
print(test_csv)

sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv" )


print(train_csv.shape)      # (10886, 11)

print(test_csv.shape)       # (6493, 8)

print(train_csv.isnull().sum()) 
print(test_csv.isna().sum())

x = train_csv.drop(['casual' , 'registered', 'count'], axis= 1 )        # [6493 rows x 8 columns] // drop을 줄 때 '를 따로 따로 줘야된다.
y = train_csv['count']

print(x)
print(y)            #  10886, 

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state= 77 )

#2 모델구성
model = Sequential()
model.add(Dense(10,input_dim = 8))
model.add(Dense(25))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))


#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer='adam')
model.fit(x_train, y_train, epochs = 3000 , batch_size= 50)

#4 평가, 예측
loss = model.evaluate(x_test,y_test)

y_submit = model.predict(test_csv)

print(y_submit)
print(y_submit.shape)       # (6493, 1)


# 결과 넣기
sampleSubmission_csv['count'] = y_submit
print(sampleSubmission_csv)             # [6493 rows x 2 columns]

sampleSubmission_csv.to_csv(path + "sampleSubmission_0108.csv" , index = False)

print("로스는 : " , loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 = " ,r2)



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