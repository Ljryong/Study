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

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2 모델 구성

#3 컴파일, 훈련
from keras.optimizers import Adam
learning_rates = [ 1.0, 0.1, 0.01, 0.001, 0.0001 ]

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss' , mode = "min" , verbose = 1 , patience = 50 , restore_best_weights=True )
# epoch 랑 patience를 같게 줘서 epoch의 최솟값을 가질수는 있겠지만, 그럴바에 epoch 를 10000000을 주고 patience 를 10000 을 주는게 더 좋을것이다.

for learning_rate in learning_rates :
    model = Sequential()
    model.add(Dense(1024, input_dim = 9 , activation= 'relu' ))
    # model.add(Dense(512,)) 
    # model.add(Dense(1024))
    # model.add(Dense(2048))
    model.add(Dense(512, activation= 'relu'))
    model.add(Dense(256, activation= 'relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer= Adam(learning_rate) , metrics = ['mse' , 'mae'])
    hist = model.fit(x_train,y_train, epochs= 1000000 ,batch_size= 10 , validation_split=0.2 , callbacks= [es])

    #4 평가, 예측
    loss = model.evaluate(x_test,y_test)

    y_predict = model.predict(x_test)
    print('lr : {0}, 로스 : {1} '.format(learning_rate,loss[0]))
    r2 = r2_score(y_test,y_predict)
    print('lr : {0} , R2 : {1} '.format(learning_rate,r2))


'''
lr : 1.0, 로스 : 25895.33984375 
lr : 1.0 , R2 : -3.3406066214175842

lr : 0.1, 로스 : 2269.505859375
lr : 0.1 , R2 : 0.6195828176244282

lr : 0.01, 로스 : 1758.8917236328125
lr : 0.01 , R2 : 0.7051725657619057

lr : 0.001, 로스 : 1829.85302734375
lr : 0.001 , R2 : 0.6932779150089079

lr : 0.0001, 로스 : 2217.800048828125
lr : 0.0001 , R2 : 0.6282498549248078
'''


# plt.figure(figsize = (9,6))
# plt.plot(hist.history['loss'], c = 'red' , marker = '.' , label = 'loss')
# plt.plot(hist.history['val_loss'], c = 'blue' , marker = '.' , label = 'val_loss')
# plt.legend(loc = 'upper right')

# plt.title('dacon loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.grid()
# plt.show()



# Epoch 678: early stopping
# 14/14 [==============================] - 0s 1ms/step - loss: 2076.9592 - mse: 2076.9592 - mae: 32.1103
# 23/23 [==============================] - 0s 848us/step
# 14/14 [==============================] - 0s 1ms/step
# R2 :  0.6883761242209259
# loss :  [2076.959228515625, 2076.959228515625, 32.11029052734375]
# RMSE :  45.57367065064922

# MinMaxScaler
# Epoch 824: early stopping
# 14/14 [==============================] - 0s 1ms/step - loss: 2116.0803 - mse: 2116.0803 - mae: 31.8133
# 23/23 [==============================] - 0s 829us/step
# 14/14 [==============================] - 0s 0s/step
# R2 :  0.6453002008465227
# loss :  [2116.080322265625, 2116.080322265625, 31.813270568847656]
# RMSE :  46.00087287015406

# StandardScaler
# Epoch 393: early stopping
# 14/14 [==============================] - 0s 2ms/step - loss: 1927.2096 - mse: 1927.2096 - mae: 31.7532
# 23/23 [==============================] - 0s 1ms/step
# 14/14 [==============================] - 0s 859us/step
# R2 :  0.6769589264272043
# loss :  [1927.2095947265625, 1927.2095947265625, 31.753246307373047]
# RMSE :  43.89999529607844

# MaxAbsScaler
# Epoch 756: early stopping
# 14/14 [==============================] - 0s 985us/step - loss: 1761.8816 - mse: 1761.8816 - mae: 30.7702
# 23/23 [==============================] - 0s 904us/step
# 14/14 [==============================] - 0s 166us/step
# R2 :  0.7046714003829079
# loss :  [1761.881591796875, 1761.881591796875, 30.770217895507812]
# RMSE :  41.97477282047839

# RobustScaler
# Epoch 529: early stopping
# 14/14 [==============================] - 0s 991us/step - loss: 2221.0278 - mse: 2221.0278 - mae: 33.6121
# 23/23 [==============================] - 0s 952us/step
# 14/14 [==============================] - 0s 960us/step
# R2 :  0.6277087989424783
# loss :  [2221.02783203125, 2221.02783203125, 33.6120719909668]
# RMSE :  47.12778052794729



