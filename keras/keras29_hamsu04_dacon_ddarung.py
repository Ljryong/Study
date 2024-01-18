from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error
from keras.models import Sequential , Model
from keras.layers import Dense , Dropout , Input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


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

date = datetime.datetime.now()
date = date.strftime('%m%d-%H%M')
path = 'c:/_data/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path , 'k28_4_', date , '_', filename ])


#2 모델 구성
# model = Sequential()
# model.add(Dense(1024, input_shape = (9,)  ))
# model.add(Dense(512))
# model.add(Dropout(0.3))
# model.add(Dense(256, activation= 'relu'))
# model.add(Dense(1))

# 2-1
input = Input(shape=(9,))
d1 = Dense(1024)(input)
d2 = Dense(512)(d1)
drop1 = Dropout(0.3)(d2)
d3 = Dense(256)(drop1)
output = Dense(1)(d3)
model = Model(inputs =input , outputs= output)


#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam' , metrics = ['mse' , 'mae'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'loss' , mode = "min" , verbose = 1 , patience = 10 , restore_best_weights=True )
# epoch 랑 patience를 같게 줘서 epoch의 최솟값을 가질수는 있겠지만, 그럴바에 epoch 를 10000000을 주고 patience 를 10000 을 주는게 더 좋을것이다.

mcp = ModelCheckpoint(monitor='val_loss' , mode = 'min' , verbose= 1 , save_best_only=True , filepath= filepath )

hist = model.fit(x_train,y_train, epochs= 1000000 ,batch_size= 10 , validation_split=0.2 , callbacks= [es,mcp])

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

# Dropout
# Epoch 26: early stopping
# 14/14 [==============================] - 0s 1ms/step - loss: 2319.9834 - mse: 2319.9834 - mae: 35.9168
# 23/23 [==============================] - 0s 813us/step
# 14/14 [==============================] - 0s 0s/step
# R2 :  0.6111217705229142
# loss :  [2319.9833984375, 2319.9833984375, 35.91681671142578]
# RMSE :  48.166203547336
