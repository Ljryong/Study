# 14_1 카피


from sklearn.datasets import load_boston
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error 
import time

# warning 뜨는것을 없애는 방법, 하지만 아직 왜 뜨는지 모르니 보는것을 추천
import warnings
warnings.filterwarnings('ignore') 

# 현재 사이킷런 버전 1.3.0 보스턴 안됨, 그래서 삭제
# pip uninstall scikit-learn
# pip uninstall scikit-image
# pip uninstall scikit-learn-intelex

# pip install scikit-learn==0.23.2




datasets = load_boston()
print(datasets)
x = datasets.data
y = datasets.target
print(x)
print(x.shape)      #(506, 13)
print(y)
print(y.shape)      #(506,)

print(datasets.feature_names)
# 'CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

print(datasets.DESCR)               
#[실습]
# train , test의 비율을 0.7 이상 0.9 이하
# R2 0.62 이상
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 51  )


#2
model = Sequential()
model.add(Dense(20,input_dim = 13))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(14))
model.add(Dense(7))
model.add(Dense(1))

#3
model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping               # keras.callbacks에서  Earlystopping 을 떙겨온다
es = EarlyStopping(monitor='val_loss',      # 
                   mode = 'min'  ,          # 모드는 최솟값으로 한다. (max,auto도 가능하다) auto는 자동으로 낮고 높음을 조절해준다. ex) loss는 낮게 r2는 높게 해준다.
                   patience= 10  ,          # 최솟값이 갱신되는걸 10번까지 봐준다.
                   verbose = 1   ,)


start_time = time.time()

hist = model.fit(x_train,y_train,epochs=800,batch_size=10,validation_split=0.2 , callbacks = [es] ) # [] 한개인데 대괄호일 때 에는 들어갈 수 있는게 더 있다고 생각해야 된다.
# loss 와 val_loss 가 들어가 있다.

end_time = time.time()

#4
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test,y_predict)

# def RMSE(a,b):
#     np.sqrt(mean_squared_error(aaa,bbb))
# rmse = RMSE(y_test,y_predict)
# print(rmse)


print('R2 : ' , r2)
print(end_time - start_time)            # python에서 기본으로 제공하는 시스템
                                        # print는 함수



#================================
print(" hist :",hist)
#================================
print(hist.history)
#================================
print(hist.history["loss"])
#================================
print(hist.history["val_loss"])
#================================

import matplotlib.pyplot as plt

plt.rcParams['font.family'] ='Malgun Gothic'  

plt.figure(figsize=(9,6))               # 그래프 창 크기
plt.plot(hist.history['loss'], c = 'red' , label = 'loss' , marker='.')
# c = 'red' , label = 'loss' , marker='.' // c = color , label = 이름 , marker = 1 epoch 당 . 을 찍어주세요
plt.plot(hist.history['val_loss'], c = 'blue' , label = 'val_loss' , marker='.')






plt.legend(loc='upper right') # 라벨을 오른쪽 위에 달아주세요
plt.title('보스턴 loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()

plt.show()










