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


train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

print(train_csv.isna().sum())
print(test_csv.isna().sum())


x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state = 9 , shuffle= True)  # 9 , 3 , 11 , 12

#2 모델 구성
model = Sequential()
model.add(Dense(1024, input_dim = 9  ))
# model.add(Dense(512,)) 
# model.add(Dense(1024))
# model.add(Dense(2048))
model.add(Dense(512))
model.add(Dense(256, activation= 'relu'))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam' , metrics = ['mse' , 'mae'])

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss' , mode = "min" , verbose = 1 , patience = 50 , restore_best_weights=True )
# epoch 랑 patience를 같게 줘서 epoch의 최솟값을 가질수는 있겠지만, 그럴바에 epoch 를 10000000을 주고 patience 를 10000 을 주는게 더 좋을것이다.

hist = model.fit(x_train,y_train, epochs= 1000000 ,batch_size= 10 , validation_split=0.2 , callbacks= [es])

#4 평가, 예측
loss = model.evaluate(x_test,y_test)

y_submit = model.predict(test_csv)

submission_csv['count'] = y_submit

submission_csv.to_csv(path + "submission_0111.csv", index = False)

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


# 14/14 [==============================] - 0s 1ms/step
# R2 :  0.518813528188014
# loss :  [3340.6865234375, 3340.6865234375, 44.581275939941406]
# RMSE :  57.79867421275639

# R2 :  0.5921279139813469
# loss :  [2754.93310546875, 2754.93310546875, 40.1713981628418]
# RMSE :  52.48745904949475


# 59.9
# Epoch 358: early stopping
# 14/14 [==============================] - 0s 1ms/step - loss: 2760.6423 - mse: 2760.6423 - mae: 36.6228
# 23/23 [==============================] - 0s 839us/step
# 14/14 [==============================] - 0s 0s/step
# R2 :  0.6353800311369304
# loss :  [2760.642333984375, 2760.642333984375, 36.62281799316406]
# RMSE :  52.54181573995065

# 71
# Epoch 176: early stopping
# 14/14 [==============================] - 0s 1ms/step - loss: 2451.4075 - mse: 2451.4075 - mae: 38.7302
# 23/23 [==============================] - 0s 771us/step
# 14/14 [==============================] - 0s 1ms/step
# R2 :  0.6518478370221634
# loss :  [2451.407470703125, 2451.407470703125, 38.73020935058594]
# RMSE :  49.51169155107232

# Epoch 141: early stopping
# 14/14 [==============================] - 0s 1ms/step - loss: 2715.8901 - mse: 2715.8901 - mae: 39.4275
# 23/23 [==============================] - 0s 501us/step
# 14/14 [==============================] - 0s 1ms/step
# R2 :  0.6258176460727904
# loss :  [2715.89013671875, 2715.89013671875, 39.4275016784668]
# RMSE :  52.114202201005995


# 64
# Epoch 551: early stopping
# 14/14 [==============================] - 0s 658us/step - loss: 2607.2590 - mse: 2607.2590 - mae: 39.1473
# 23/23 [==============================] - 0s 780us/step
# 14/14 [==============================] - 0s 1ms/step
# R2 :  0.6170046316388503
# loss :  [2607.259033203125, 2607.259033203125, 39.14725112915039]
# RMSE :  51.061326963866826

# Epoch 499: early stopping
# 14/14 [==============================] - 0s 0s/step - loss: 2100.0142 - mse: 2100.0142 - mae: 32.7952
# 23/23 [==============================] - 0s 870us/step
# 14/14 [==============================] - 0s 1ms/step
# R2 :  0.684917016342975
# loss :  [2100.01416015625, 2100.01416015625, 32.7951545715332]
# RMSE :  45.825912525325364

# Epoch 678: early stopping
# 14/14 [==============================] - 0s 1ms/step - loss: 2076.9592 - mse: 2076.9592 - mae: 32.1103
# 23/23 [==============================] - 0s 848us/step
# 14/14 [==============================] - 0s 1ms/step
# R2 :  0.6883761242209259
# loss :  [2076.959228515625, 2076.959228515625, 32.11029052734375]
# RMSE :  45.57367065064922













