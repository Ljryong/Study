from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

#1 데이터

path = "c:/_data/dacon/ddarung//"

train_csv = pd.read_csv(path +"train.csv", index_col= 0  )
test_csv = pd.read_csv(path +"test.csv", index_col= 0  )
submission_csv = pd.read_csv(path +"submission.csv"  )

print(train_csv)        # [1459 rows x 10 columns]
print(test_csv)         # [715 rows x 9 columns]
print(submission_csv)   # [715 rows x 2 columns]

print(train_csv.columns)            # ['hour', 'hour_bef_temperature', 'hour_bef_precipitation', 
                                    #   'hour_bef_windspeed','hour_bef_humidity', 'hour_bef_visibility',
                                    #   'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count']

print(train_csv.isna().sum())
train_csv = train_csv.fillna(0)
test_csv = test_csv.fillna(0)

# x = train_csv.drop(['count'],axis = 1)s
# y = train_csv(['count'])


x = train_csv.drop(['count'],axis = 1)   
print(x)
y = train_csv['count']                   



x_train , x_test , y_train, y_test = train_test_split(x,y,test_size = 0.3 , random_state= 514 , shuffle= True)

#2 모델구성
model = Sequential()
model.add(Dense(13,input_dim = 9))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))



#3 컴파일, 훈련
model.compile(loss='mse',optimizer = 'adam')
model.fit(x_test,y_test,epochs = 1500, batch_size = 5 , validation_split=0.3)

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_submit = model.predict(test_csv)

submission_csv['count'] = y_submit
submission_csv.to_csv(path + "submission_0109.csv",index=False)

print("로스는 : " , loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print("R2 = " ,r2)


# Epoch 1000/1000
# 35/35 [==============================] - 0s 2ms/step - loss: 2687.4595 - val_loss: 2807.8896
# 14/14 [==============================] - 0s 908us/step - loss: 2622.1301
# 23/23 [==============================] - 0s 835us/step
# 로스는 :  2622.130126953125
# 14/14 [==============================] - 0s 0s/step
# R2 =  0.6404279786579505