import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1
path = "c:/_data/kaggle/bike//"
train_csv = pd.read_csv(path + 'train.csv' , index_col = 0 )
test_csv = pd.read_csv(path + "test.csv" , index_col = 0 )
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
print(train_csv)        # [10886 rows x 11 columns]
print(test_csv)         # [6493 rows x 8 columns]
print(submission_csv)   # [6493 rows x 2 columns]

print(train_csv.info())
print(test_csv.info())
print(submission_csv.info())

x = train_csv.drop(['casual' , 'registered' , 'count'] , axis= 1)
y = train_csv['count']        
print(x)       
print(y)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size= 0.3 , random_state= 0 )

#2 모델구성
model = Sequential()#클래스 객체화
model.add(Dense(1,input_dim = 8, activation= 'relu' ))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs = 100 ,batch_size= 100 , validation = 0.2)

#4 평가 , 예측
loss= model.evaluate(x_test,y_test)
y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit          # 소괄호는 안되고 대괄호만 가능하다. ---> 함수 뒤에만 

print(submission_csv)

# submission_csv = submission_csv.to_csv(path + 'submission_0108.csv' , index = False)
