from keras.models import Sequential , Model
from keras.layers import Dense , Dropout , Input , Conv2D , Flatten , MaxPooling1D , LSTM , Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score , mean_squared_error
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import time

#1 데이터
path = "c:/_data/dacon/diabetes//"


train_csv = pd.read_csv(path + "train.csv",index_col = 0)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
submission_csv = pd.read_csv(path + "sample_submission.csv")

print(train_csv)        # [652 rows x 9 columns]
print(test_csv)         # [116 rows x 8 columns]


print(train_csv.isna().sum())
print(test_csv.isna().sum())

x = train_csv.drop(['Outcome'],axis = 1)            # 독립변수 = x , 종속변수 = y 
y = train_csv['Outcome']

print(x)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 22 , shuffle=True )  # 7 98 22 
es = EarlyStopping(monitor= 'val_loss' , mode = "min" , verbose= 1 , patience= 10 , restore_best_weights= True)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

# print(x_train.shape)       # (456, 8)
# print(x_test.shape)        # (196, 8)
# print(test_csv.shape)      # (116, 8)


x_train = x_train.values.reshape(456,4,2)
x_test = x_test.values.reshape(196,4,2)
test_csv = test_csv.values.reshape(116,4,2)





###################
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

date = datetime.datetime.now()
date = date.strftime('%m%d-%H%M')
path = 'c:/_data/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path , 'k28_7_', date , '_', filename ])


#2 모델구성
# model = Sequential()
# model.add(Dense(64,input_shape = (8,)))
# model.add(Dense(128))
# model.add(Dropout(0.3))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(1,activation='sigmoid'))

#2-1
# input = Input(shape = (8,))
# d1 = Dense(64)(input)
# d2 = Dense(128)(d1)
# drop1 = Dropout(0.3)(d2)
# d3 = Dense(64)(drop1)
# d4 = Dense(32)(d3)
# d5 = Dense(16)(d4)
# d6 = Dense(8)(d5)
# output = Dense(1,activation='sigmoid')(d6)
# model = Model(inputs = input , outputs = output)

# 2-2
model = Sequential()
model.add(Conv1D(64,2,input_shape = (4,2) ))
model.add(Flatten())
model.add(Dense(36,activation='relu'))
model.add(Dense(1,activation='sigmoid'))



mcp = ModelCheckpoint(monitor='val_loss', mode='min' , verbose=1, save_best_only=True , filepath=  filepath   )

#3 컴파일, 훈련
model.compile(loss = 'binary_crossentropy' , optimizer= 'adam' , metrics=['acc'])
start_time = time.time()
model.fit(x_train,y_train,epochs = 1000 , batch_size = 100 , validation_split=0.2 , callbacks = [es,mcp] )
end_time = time.time()

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_submit = np.round(model.predict(test_csv))

submission_csv['Outcome'] = y_submit
y_predict = model.predict(x_test)
submission_csv.to_csv(path + 'submission_0110.csv', index=False)


print(y_test)
print(y_predict)

def ACC(aaa, bbb):
    return accuracy_score(aaa,bbb)

acc = ACC(y_test, np.round(y_predict))
print('loss = ',loss)
print("Acc = ",acc)
print('시간 : ',end_time - start_time)


# Acc =  0.45782327521923516

# Acc =  0.46007830505021624

# Acc =  0.4701709538327378

# Acc =  0.47333580690547894

# Acc =  0.6989795918367347

# Acc =  0.7244897959183674

# Acc =  0.7908163265306123

# Acc =  0.7959183673469388


# loss =  [0.488598495721817, 0.795918345451355]
# Acc =  0.7959183673469388


# MinMaxScaler
# loss =  [0.5009446740150452, 0.7551020383834839]
# Acc =  0.7551020408163265

# StandardScaler
# loss =  [0.4931170344352722, 0.7857142686843872]
# Acc =  0.7857142857142857

# MaxAbsScaler
# loss =  [0.4855547249317169, 0.7908163070678711]
# Acc =  0.7908163265306123

# RobustScaler
# loss =  [0.5026631951332092, 0.75]
# Acc =  0.75

# Dropout
# loss =  [0.5997169017791748, 0.7142857313156128]
# Acc =  0.7142857142857143



# cpu
# 시간 :  21.6106219291687
# gpu
# 시간 :  24.430900812149048


# Cnn
# loss =  [0.5519381761550903, 0.7346938848495483]
# Acc =  0.7346938775510204
# 시간 :  4.619797468185425

# Conv1D
# loss =  [0.6207653880119324, 0.6632652878761292]
# Acc =  0.6632653061224489
# 시간 :  3.481691598892212
