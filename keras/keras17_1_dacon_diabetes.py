from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score , mean_squared_error
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping

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

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 444 , shuffle=True )  # 7 98 22 
es = EarlyStopping(monitor= 'val_loss' , mode = "min" , verbose= 1 , patience= 300 , restore_best_weights= True)

#2 모델구성
model = Sequential()
model.add(Dense(64,input_dim = 8))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1,activation='sigmoid'))



#3 컴파일, 훈련
model.compile(loss = 'binary_crossentropy' , optimizer= 'adam' , metrics=['acc'])
model.fit(x_train,y_train,epochs = 1000000 , batch_size = 5 , validation_split=0.2 , callbacks = [es] )

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

print("Acc = ",acc)


# Acc =  0.45782327521923516

# Acc =  0.46007830505021624

# Acc =  0.4701709538327378

# Acc =  0.47333580690547894

# Acc =  0.6989795918367347

# Acc =  0.7244897959183674

# Acc =  0.7908163265306123

# Acc =  0.7959183673469388