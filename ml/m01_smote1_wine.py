import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import PolynomialFeatures


#1 데이터

############ 불균형한 데이터가 없어서 만듬 ############
datasets = load_wine()
x= datasets.data
y= datasets.target      # = y= datasets.['target'] 이랑 같다.

print(x.shape,y.shape)      # (178, 13) (178,)
print(np.unique(y,return_counts=True))      # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.value_counts(y))
print(y)

x =x[:-35]
y =y[:-35]
print(y)
print(np.unique(y,return_counts=True))      # (array([0, 1, 2]), array([59, 71,  8], dtype=int64))

############ 불균형한 데이터가 없어서 만듬 ############


x_train, x_test , y_train , y_test = train_test_split(x,y, test_size=0.25 , random_state= 156 , stratify=y , shuffle=True )

from keras.models import Sequential
from keras.layers import Dense

'''
#2 모델구성
model = Sequential()
model.add(Dense(10,input_shape = (13,)))
model.add(Dense(3, activation='softmax'))

#3 컴파일, 훈련
model.compile(loss = 'sparse_categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train , epochs= 10, validation_batch_size=0.2)

# 'sparce_categorical_crossentropy' = 다중분류에서 OneHot을 사용하기 싫을 때 쓴다. 자동으로 OneHot이 들어가있음


#4 평가, 예측
result = model.evaluate(x_test,y_test)
print('loss',result[0])
print('acc',result[1])


y_predict = model.predict(x_test)



# print(y_test)           # 원핫 안되어 잇음
# print(y_predict)        # 원핫 되어있음

y_predict = np.argmax(y_predict,axis=1)

# print(y_predict)        # 원핫 안되게 바뀜


print(f1_score(y_test, y_predict , average='macro' ))

# f1 = acc를 보충한 지표 (2진분류용으로 만들어짐)
# 라벨이 하나가 크고 하나가 작을때 사용한다. 
# 이 때 acc보다 정확한 정보를 전달한다.




# Epoch 100/100
# 4/4 [==============================] - 0s 997us/step - loss: 0.3227 - acc: 0.9099
# 2/2 [==============================] - 0s 999us/step - loss: 0.5943 - acc: 0.7297
# loss 0.5943061709403992
# acc 0.7297297120094299



# loss 0.6638224124908447
# acc 0.7777777910232544
# 2/2 [==============================] - 0s 996us/step
# 0.7273378950798306





# loss 407.3161926269531
# acc 0.5
# 0.2222222222222222

'''



########## smote ########### 
print('smote 적용')

from imblearn.over_sampling import SMOTE        # 옛날에는 다운을 받았어야 됬는데 anaconda가 다 끌어옴
import sklearn as sk
print('사이킷 런 :' , sk.__version__)

smote = SMOTE(random_state=0)
x_train , y_train = smote.fit_resample(x_train,y_train)                    # 다시 샘플링한다

print(pd.value_counts(y_train))                    # 0    53 , 1    53 , 2    53    // 53인 이유는 train과 test로 짤려서 그렇다.
print(np.unique(x_train,return_counts=True))


#2 모델구성
model = Sequential()
model.add(Dense(10,input_shape = (13,)))
model.add(Dense(3, activation='softmax'))

#3 컴파일, 훈련
model.compile(loss = 'sparse_categorical_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train,y_train , epochs= 1000, validation_batch_size=0.2 )

# 'sparce_categorical_crossentropy' = 다중분류에서 OneHot을 사용하기 싫을 때 쓴다. 자동으로 OneHot이 들어가있음


#4 평가, 예측
result = model.evaluate(x_test,y_test)
print('loss',result[0])
print('acc',result[1])


y_predict = model.predict(x_test)



# print(y_test)           # 원핫 안되어 잇음
# print(y_predict)        # 원핫 되어있음

y_predict = np.argmax(y_predict,axis=1)

# print(y_predict)        # 원핫 안되게 바뀜


print(f1_score(y_test, y_predict , average='macro' ))

# Epoch 10/10
# 5/5 [==============================] - 0s 997us/step - loss: 5.1487 - acc: 0.3270
# 2/2 [==============================] - 0s 2ms/step - loss: 3.6503 - acc: 0.5833
# loss 3.6503305435180664
# acc 0.5833333134651184
# 2/2 [==============================] - 0s 0s/step
# 0.47398736529171304