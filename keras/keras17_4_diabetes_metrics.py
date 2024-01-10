from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import time
import matplotlib.pyplot as plt

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)          # (442, 10) (442,)
print(datasets.feature_names)   #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 151235 , shuffle= True )

#2 모델구성
model = Sequential()
model.add(Dense(50,input_dim = 10 ))
model.add(Dense(25))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer = 'adam' , metrics = ['mse' , 'mae'])

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss' , mode = 'min' , patience = 10, verbose=1 , restore_best_weights=True ) 

hist = model.fit(x_train,y_train,epochs = 10000 , batch_size = 5 , validation_split = 0.2 )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)



plt.figure(figsize = (9,6))
plt.plot(hist.history['loss'], c = 'red' , label = 'loss' , marker = '.')
plt.plot(hist.history['val_loss'],c = 'blue' , label = 'val_loss' , marker = '.')
plt.legend(loc = 'upper right')


print(hist)
plt.title('diabetes loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()

plt.show()


print(r2)
print(loss)

# 0.4282132267148565 = r2 
# 3493.03271484375 = loss

