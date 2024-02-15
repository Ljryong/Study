# m31에 뽑은 4가지 결과로 4가지 모델 만들기

# 1.  70000,154
# 2.  70000,331
# 3.  70000,486
# 4.  70000,713
# 5.  70000,784 원본

# 시간과 성능 체크 

# 결과 예시
# 결과1 PCA = 154
# 걸린시간 0000초
# acc = 0.0000

from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.model_selection import train_test_split

(x_train, y_train ) , (x_test, y_test ) = mnist.load_data()
# 받고 싶지 않은게 있을 때 _를 넣어주면 됨(에러가 뜨지 않음)
print(x_train.shape,x_test.shape)           # (60000, 28, 28) (10000, 28, 28)

# x = np.append(x_train,x_test,axis=0)
# x = np.concatenate([x_train , x_test],axis=0)       # 지금 사용한 append 와 concatenate는 같다
# print(x.shape)      # (70000, 28, 28)

print(np.unique(y_train))

print(x_train.shape)


# x = x.reshape(-1,x.shape[1]*x.shape[2])

x_train = x_train.reshape(x_train.shape[0] , x_train.shape[1]*x_train.shape[2] )
x_test = x_test.reshape(x_test.shape[0] , x_test.shape[1]*x_test.shape[2] )
# x_train = x_train.reshape( -1 , 784)

# print(x_train.shape)

# number = [154,331,486,713,784]
# for i,number in enumerate(number):
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=9)
x_train2=lda.fit_transform(x_train,y_train)
x_test2 = lda.transform(x_test)
evr = lda.explained_variance_ratio_
print(x_train2.shape)
print(x_test2.shape)
# pca_cumsum = np.cumsum(evr)
# print( 'n_components:',784 ,'변화율', pca_cumsum)

#2 데이터
model = Sequential()
model.add(Dense(10,input_shape = (9,)))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='softmax'))

#3 컴파일, 훈련
model.compile(loss = 'sparse_categorical_crossentropy' , optimizer='adam' , metrics=['acc']  )
start = time.time()
model.fit(x_train2,y_train,epochs= 100 , batch_size=500 , validation_split=0.2   )
end = time.time()

#4 평가, 예측
result = model.evaluate(x_test2,y_test)

print('걸린 시간',end-start)
print('acc',result[1])


# 결과 1 PCA= 154
# 걸린 시간 30.1601459980011
# acc 0.9391000270843506

# 결과 2 PCA= 331
# 걸린 시간 31.5673885345459
# acc 0.9144999980926514

# 결과 3 PCA= 486
# 걸린 시간 33.23084259033203
# acc 0.9117000102996826

# 결과 4 PCA= 713
# 걸린 시간 35.7853729724884
# acc 0.911899983882904

# 결과 5 PCA= 784
# 걸린 시간 36.44583511352539
# acc 0.9103000164031982


