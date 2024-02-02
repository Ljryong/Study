from keras.datasets import imdb
import numpy as np


# imdb 영화 리뷰 25000개를 가지고 긍정인지 부정인지 판단


#1 데이터 전처리
(x_train,y_train) , (x_test,y_test) =imdb.load_data(num_words=240)


# print(x_train.shape,y_train.shape)      # (25000,) (25000,)
# print(x_test.shape,y_test.shape)        # (25000,) (25000,)
# print(len(x_train[0]),len(x_test[0]))   # 218 68
# print(y_train[:20])
# print(np.unique(y_train,return_counts=True))        # (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64)) = 0,1 이진분류


# print(sum(map(len,x_train))/len(x_train))           # 238.71364 평균 값

# print(max(len(i) for i in x_train))                 # 2494 최대 길이 값

from keras.utils import pad_sequences
x_train = pad_sequences(x_train, maxlen=240 , truncating='pre' ,padding='pre'  )
x_test = pad_sequences(x_test ,maxlen=240 , truncating='pre' ,padding='pre'  )

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss' , mode = 'min' , patience= 100 , restore_best_weights=True , verbose= 1  )









#2 모델구성
from keras.models import Sequential
from keras.layers import Dense , LSTM , Flatten , Conv1D , Embedding , MaxPooling1D
model = Sequential()
model.add(Embedding(240,80,input_length=240))
model.add(MaxPooling1D())
model.add(Conv1D(16,2,activation='relu'))
model.add(Conv1D(64,2,activation='relu'))
model.add(Conv1D(16,2,activation='relu'))
model.add(Conv1D(64,2,activation='relu'))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

'''

#3 컴파일, 훈련
model.compile(loss = 'binary_crossentropy' , optimizer='adam' , metrics=['acc'] )
model.fit(x_train, y_train , epochs= 100000 , batch_size= 1000 , validation_split= 0.2 , callbacks=[es])

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
pre = model.predict(x_test)

from sklearn.metrics import accuracy_score
print(loss)


y_test = np.round(y_test)
pre = np.round(pre)

print(accuracy_score(y_test,pre))


# [0.45681604743003845, 0.7844799757003784]
# 0.78448
'''