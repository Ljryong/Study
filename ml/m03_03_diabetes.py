from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score
from keras.models import Sequential , load_model
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

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2 모델구성
from sklearn.svm import LinearSVR

from sklearn.linear_model import Perceptron , LogisticRegression , LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
models = [LinearSVR(),Perceptron(),LogisticRegression(),RandomForestClassifier(),DecisionTreeClassifier(),KNeighborsClassifier()]



#3 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy' , 'mse', 'mae'])     
# binary_crossentropy = 2진 분류 = y 가 2개면 무조건 이걸 사용한다
# accuracy = 정확도 = acc
# metrics로 훈련되는걸 볼 수는 있지만 가중치에 영향을 미치진 않는다.

# metrics가 accuracy 

############## 훈련 반복 for 문 ###################
for model in models :
    model.fit(x_train,y_train)
    result = model.score(x_test,y_test)
    print(f'{type(model).__name__} score ',result)
    y_predict = model.predict(x_test)
    print(f'{type(model).__name__} predict ',r2_score(y_test,y_predict))
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



# 0.4375280766636177
# 0.4375280766636177


# LinearSVR score  -0.11584717273232203
# LinearSVR predict  -0.11584717273232203
# Perceptron score  0.0
# Perceptron predict  -0.17001095003895728
# LogisticRegression score  0.0
# LogisticRegression predict  0.2471293071081715
# RandomForestClassifier score  0.007518796992481203
# RandomForestClassifier predict  -0.05620089357085534
# DecisionTreeClassifier score  0.0
# DecisionTreeClassifier predict  -0.11524009113458811
# KNeighborsClassifier score  0.007518796992481203
# KNeighborsClassifier predict  -0.9171868323838201