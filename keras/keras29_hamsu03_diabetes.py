from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential , Model
from keras.layers import Dense , Dropout , Input
import numpy as np
import time
import matplotlib.pyplot as plt
import datetime


#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)          # (442, 10) (442,)
print(datasets.feature_names)   #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 151235 , shuffle= True )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

date = datetime.datetime.now()
date = date.strftime('%m%d-%H%M')
path = 'c:/_data/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path , 'k28_3_', date , '_', filename ])



# #2 모델구성
# model = Sequential()
# model.add(Dense(50,input_shape = 10 ))
# model.add(Dropout(0.2))
# model.add(Dense(25))
# model.add(Dense(1))

#2-1
input = Input(shape = (10,))
d1 = Dense(50)(input)
drop1 = Dropout(0.2)(d1)
d2 = Dense(25)(drop1)
output = Dense(1)(d2)

model = Model(inputs = input , outputs = output)

#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer = 'adam' , metrics = ['mse' , 'mae'])

from keras.callbacks import EarlyStopping , ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss' , mode = 'min' , patience = 10, verbose=1 , restore_best_weights=True ) 

mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'min' , verbose = 1 , save_best_only=True , filepath= filepath )

hist = model.fit(x_train,y_train,epochs = 100000000000 , batch_size = 5 , validation_split = 0.2 ,callbacks= [es,mcp] )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)



# plt.figure(figsize = (9,6))
# plt.plot(hist.history['loss'], c = 'red' , label = 'loss' , marker = '.')
# plt.plot(hist.history['val_loss'],c = 'blue' , label = 'val_loss' , marker = '.')
# plt.legend(loc = 'upper right')


# print(hist)
# plt.title('diabetes loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()

# plt.show()


print(r2)
print(loss)

# 0.4282132267148565 = r2 
# 3493.03271484375 = loss


# MinMaxScaler
# Epoch 132: early stopping
# 5/5 [==============================] - 0s 0s/step - loss: 3495.9897 - mse: 3495.9897 - mae: 48.7478
# 5/5 [==============================] - 0s 0s/step
# 0.4277291969179722
# [3495.98974609375, 3495.98974609375, 48.7478141784668]

# StandardScaler
# Epoch 114: early stopping
# 5/5 [==============================] - 0s 1ms/step - loss: 3423.5649 - mse: 3423.5649 - mae: 47.5210
# 5/5 [==============================] - 0s 0s/step
# 0.43958468538922235
# [3423.56494140625, 3423.56494140625, 47.52100372314453]

# MaxAbsScaler
# Epoch 120: early stopping
# 5/5 [==============================] - 0s 0s/step - loss: 3412.5415 - mse: 3412.5415 - mae: 47.4617
# 5/5 [==============================] - 0s 0s/step
# 0.44138915418500335
# [3412.54150390625, 3412.54150390625, 47.46168899536133]

# RobustScaler
# Epoch 144: early stopping
# 5/5 [==============================] - 0s 4ms/step - loss: 3471.2327 - mse: 3471.2327 - mae: 47.7108
# 5/5 [==============================] - 0s 0s/step
# 0.4317817805060158
# [3471.232666015625, 3471.232666015625, 47.71084976196289]


# Dropout
# Epoch 36: val_loss did not improve from 4040.04980
# 50/50 [==============================] - 0s 997us/step - loss: 2463.5974 - mse: 2463.5974 - mae: 40.5088 - val_loss: 4116.0996 - val_mse: 4116.0996 - val_mae: 52.2245
# Epoch 36: early stopping
# 5/5 [==============================] - 0s 0s/step - loss: 3421.3333 - mse: 3421.3333 - mae: 47.4588
# 5/5 [==============================] - 0s 499us/step
# 0.4399500408441176
# [3421.333251953125, 3421.333251953125, 47.45878601074219]

