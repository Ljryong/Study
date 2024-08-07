import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score , mean_squared_error , mean_squared_log_error , accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


#1 데이터                     ####################################      2진 분류        ###########################################
datasets = load_breast_cancer()
# print(datasets)     
print(datasets.DESCR) # datasets 에 있는 describt 만 뽑아줘
print(datasets.feature_names)       # 컬럼 명들이 나옴

x = datasets.data       # x,y 를 정하는 것은 print로 뽑고 data의 이름과 target 이름을 확인해서 적는 것
y = datasets.target
print(x.shape,y.shape)  # (569, 30) (569,)

es = EarlyStopping(monitor='val_loss' , mode = 'min' , verbose= 1 , patience= 100 ,restore_best_weights=True )

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 0 ,shuffle=True) # 0

print(np.unique(y)) # [0 1]




# y 안에 있는 array와 해당 array의 개수들을 알 수 있다.
# numpy 방법
print(np.unique(y, return_counts=True))              # (array([0, 1]), array([212, 357], dtype=int64)) //  0은 212개 1은 357개 


# pandas 방법
print(pd.DataFrame(y).value_counts())               # 3개 다 같지만 행렬일 경우 맨 위에것을 쓰고 아닐경우는 통상적으로 짧은 2번째를 쓴다.
print(pd.value_counts(y))                           
print(pd.Series(y).value_counts())




#2 모델구성
model = Sequential()
model.add(Dense(1024 ,input_dim = 30))      # sigmoid = 0과 1 사이에서 값이 나온다. 0.5 이상이며 1이 되고 0.5미만이면 0이된다.
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(1, activation= 'sigmoid'))  # 2진분류 에서는 loss = binary_crossentropy , 
                                            # activation ='sigmoid'최종 레이어에 써야한다.
                                            # sigmoid는 중간에도 쓸 수 있고 회귀모델에서도 쓸 수 있다.

#3 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy' , 'mse', 'mae'])     
# binary_crossentropy = 2진 분류 = y 가 2개면 무조건 이걸 사용한다
# accuracy = 정확도 = acc
# metrics로 훈련되는걸 볼 수는 있지만 가중치에 영향을 미치진 않는다.

# metrics가 accuracy 

hist = model.fit(x_train , y_train,epochs = 1000000 , batch_size = 1 ,  validation_split= 0.2 , callbacks=[es] ,)

#4 평가, 예측
loss = model.evaluate(x_test,y_test)        # evaluate = predict로 훈련한 x_test 값을 y_test 값이랑 비교하여 평가한다.
y_predict = model.predict(x_test)
r2 = r2_score(y_test , y_predict)
result = model.predict(x)

print(y_test)
print(np.round(y_predict))                  # round = 반올림 시켜주는 것


def ACC(aaa, bbb) :                                  # aaa,bbb 가 값이 들어가 있는 것이 아니라 '빈 박스' 같은 느낌이다.
    return np.sqrt(mean_squared_error(aaa, bbb))     
acc = ACC(y_test,y_predict)                          # 빈 박스를 여기 ACC() 로 묶어주고 빈 박스의 이름을 정해준 것이다.
print("ACC : ", acc)




'''
def RMSLE(y_test, y_predict) :                                 
    return np.sqrt(mean_squared_log_error(y_test , y_predict))
rmsle = RMSLE(y_test, y_predict)
print("RMSLE : " , rmsle )






plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c = 'red' , label = 'loss' , marker='.')
# c = 'red' , label = 'loss' , marker='.' // c = color , label = 이름 , marker = 1 epoch 당 . 을 찍어주세요
plt.plot(hist.history['val_loss'], c = 'blue' , label = 'val_loss' , marker='.')

plt.plot(hist.history['accuracy'], c = 'green' , label = 'accuracy' , marker = '.')

plt.legend(loc='upper right') # 라벨을 오른쪽 위에 달아주세요
plt.title('boston loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()

plt.show()

print("loss = ",loss)
print('r2 = ', r2)
'''

# Epoch 21: early stopping
# 6/6 [==============================] - 0s 337us/step - loss: 0.1925 - accuracy: 0.9532 - mse: 0.0526 - mae: 0.1171
# 6/6 [==============================] - 0s 555us/step
# loss =  [0.19248037040233612, 0.9532163739204407, 0.052645422518253326, 0.11708547174930573]
# r2 =  0.773750019420979

# Epoch 100: early stopping
# 6/6 [==============================] - 0s 1ms/step - loss: 0.2468 - accuracy: 0.9123 - mse: 0.0724 - mae: 0.0954
# 6/6 [==============================] - 0s 0s/step
# loss =  [0.2468472272157669, 0.9122806787490845, 0.07242728769779205, 0.09540403634309769]
# r2 =  0.6960611137712087

# Epoch 196: early stopping
# 6/6 [==============================] - 0s 1ms/step - loss: 0.1194 - accuracy: 0.9532 - mse: 0.0359 - mae: 0.0688
# 6/6 [==============================] - 0s 3ms/step
# loss =  [0.11937375366687775, 0.9532163739204407, 0.035903919488191605, 0.0688357949256897]
# r2 =  0.8549507488147026


















