from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import time                                 # 시간에 대한 정보를 가져온다

#1
datasets = fetch_california_housing()
print(datasets.items())
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape,y.shape)              # x.shape = (20640, 8) y.shpae = (20640,)

print(type(x), type(y))



print(datasets.feature_names)       #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'] // feature_names = 특징 이름 
print(datasets.DESCR)               # datasets에 대한 설명

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 59 )

#2
model = Sequential()
# model.add(Dense(13,input_dim = 8))
model.add(Dense(30, input_shape = (8,) ))       # shape으로 차원을 늘려서 input이 스칼라가 아닌 벡터로 들어감
                                                # 행을 무시하고 넣어줘야된다. 
                                                # (1561,10,10) 이면 input_shape = (10,10) 으로 넣어주면 된다.
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(14))
model.add(Dense(7))
model.add(Dense(1))

#3
model.compile(loss='mae',optimizer='adam')
start_time = time.time()                                    # 시작 시간을 기록
model.fit(x_train,y_train,epochs = 3000 , batch_size = 100)
end_time = time.time()                                      # 끝나는 시간을 기록

#4
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)
print('R2 : ',r2)
print(loss)
print('걸린시간 : ', end_time - start_time)

# R2 0.55 ~ 0.6 이상

# 145/145 [==============================] - 0s 488us/step - loss: 0.5743
# 194/194 [==============================] - 0s 425us/step - loss: 0.5125
# 194/194 [==============================] - 0s 435us/step
# R2 :  0.6143151723052073
# 걸린시간 :  328.59512186050415
# epochs = 3000 , batch_size = 100 , test_size = 0.3 , 13,30,50,30,14,7,1 , random_size = ?


# mse
# R2 :  0.6070084257218801
# 걸린시간 :  269.12228894233704
# random_state = 59


# mae
# 145/145 [==============================] - 0s 570us/step - loss: 0.5390
# 194/194 [==============================] - 0s 437us/step - loss: 0.5180
# 194/194 [==============================] - 0s 388us/step
# R2 :  0.5776539345065924
# 걸린시간 :  273.5317852497101

# R2 :  0.5718224996809571
# 걸린시간 :  280.1116874217987

# 145/145 [==============================] - 0s 586us/step - loss: 0.5358
# 194/194 [==============================] - 0s 412us/step - loss: 0.5138
# 194/194 [==============================] - 0s 413us/step
# R2 :  0.6007797449416794
# 0.5138373374938965
# 걸린시간 :  256.96805143356323


# 73/73 [==============================] - 0s 440us/step - loss: 0.5391
# 194/194 [==============================] - 0s 450us/step - loss: 0.5176
# 194/194 [==============================] - 0s 397us/step
# R2 :  0.5837401913957216
# 0.5176324248313904
# 걸린시간 :  131.3535294532776
# batch = 200