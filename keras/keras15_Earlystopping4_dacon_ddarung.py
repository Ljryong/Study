from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#1
path = "c:/_data/dacon/ddarung//"

train_csv = pd.read_csv( path + "train.csv" , index_col = 0)
test_csv = pd.read_csv( path + "test.csv" , index_col = 0)
submission_csv = pd.read_csv(path + "submission.csv")

print(train_csv)    # [1459 rows x 10 columns]
print(test_csv)     # [715 rows x 9 columns]


train_csv = train_csv.fillna(0)
test_csv = test_csv.fillna(test_csv.mean())

print(train_csv.isna().sum())
print(test_csv.isna().sum())


x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state = 35 , shuffle= True)

#2 모델 구성
model = Sequential()
model.add(Dense(50,input_dim = 9))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(400))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss' , mode = "min" , verbose = 1 , patience = 30 , restore_best_weights=True )
# epoch 랑 patience를 같게 줘서 epoch의 최솟값을 가질수는 있겠지만, 그럴바에 epoch 를 10000000을 주고 patience 를 10000 을 주는게 더 좋을것이다.

hist = model.fit(x_train,y_train, epochs= 10000 ,batch_size= 1 , validation_split=0.2 , callbacks= [es])

#4 평가, 예측
loss = model.evaluate(x_test,y_test)

y_submit = model.predict(test_csv)

submission_csv['count'] = y_submit

submission_csv.to_csv(path + "submission_0110.csv", index = False)

y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

print("R2 : " , r2)
print("loss : " , loss)

def RMSE(y_test, y_predict) :                                          
    return np.sqrt(mean_squared_error(y_test , y_predict))             
rmse = RMSE(y_test,y_predict)    
print("RMSE : ", rmse)

plt.figure(figsize = (9,6))
plt.plot(hist.history['loss'], c = 'red' , marker = '.' , label = 'loss')
plt.plot(hist.history['val_loss'], c = 'blue' , marker = '.' , label = 'val_loss')
plt.legend(loc = 'upper right')

plt.title('dacon loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()


# 로스는 :  2652.173583984375
# 14/14 [==============================] - 0s 216us/step
# R2 =  0.6363080952424065

# Epoch 27: early stopping
# 14/14 [==============================] - 0s 537us/step - loss: 2859.1780
# 23/23 [==============================] - 0s 683us/step
# 14/14 [==============================] - 0s 690us/step
# R2 :  0.607921657477723
# loss :  2859.177978515625
# RMSE :  53.47128218361674


# Epoch 80: early stopping
# 14/14 [==============================] - 0s 0s/step - loss: 2644.2185
# 23/23 [==============================] - 0s 1ms/step
# 14/14 [==============================] - 0s 2ms/step
# R2 :  0.6188786055958153
# loss :  2644.218505859375
# RMSE :  51.421967798042196


# Epoch 113: early stopping
# 14/14 [==============================] - 0s 0s/step - loss: 2544.8044
# 23/23 [==============================] - 0s 712us/step
# 14/14 [==============================] - 0s 1ms/step
# R2 :  0.6114097304456523
# loss :  2544.804443359375
# RMSE :  50.44605297893395


