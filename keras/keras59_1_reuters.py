from keras.datasets import reuters
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer

(x_train , y_train) , (x_test, y_test) = reuters.load_data(num_words=100,
                                                        #    test_split=0.3,      # train_test_split 나누는 거랑 똑같음
                                                           )


word_index = reuters.get_word_index()       # 단어의 숫자를 뽑아내는 것
정룡이는 못생겼다
print(len(np.unique(x_train))) 

print(x_train)
print(x_train.shape,x_test.shape)           # (8982,) (2246,)
print(y_train.shape,y_test.shape)           # (8982,) (2246,)
print(type(x_train))                        # <class 'numpy.ndarray'>
print(y_train)                              # [ 3  4  3 ... 25  3 25]
print(len(np.unique(y_train)))              # 46    최종 노드의 갯수
print(len(np.unique(y_test)))               # 46

print(type(x_train[0]))                     # <class 'list'>
print(len(x_train[0]))                      # 87
print(len(x_train[1]))                      # 56            길이가 다 다르다.

print('뉴스기사의 최대길이', max(len(i) for i in x_train))         # 2376          최대 길이를 뽑아내는 법
print('뉴스기사의 평균길이', sum(map(len,x_train)) / len(x_train)) # 145.5398574927633          최대 길이를 뽑아내는 법


#1 전처리
from keras.utils import pad_sequences

x_train = pad_sequences(x_train,padding='pre', maxlen=100 , truncating='pre')   # maxlen 을 100 으로 설정한 이유는 2376개를 maxlen 으로 두면 onehot을 한거랑 다를게 없고
                                                                                # 훈련할수록 0에 수렴을 해서 평균값을 찾아 145를 알아낸 뒤 그냥 100으로 잡은 것
x_test = pad_sequences(x_test,padding='pre', maxlen=100 , truncating='pre')

# y 원핫은 하고 싶으면 하고 하기 싫으면 loss = sparse_categorical_crossentropy를 쓰면 된다.

print(x_train.shape,x_test.shape)           # (8982, 100) (2246, 100)

# 만들기
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense , Conv1D ,Embedding , LSTM , Flatten
from sklearn.metrics import accuracy_score

es = EarlyStopping(monitor='val_loss' , mode= 'min' ,patience= 100 , restore_best_weights=True , verbose= 1  )

#2 모델구성
model = Sequential()
model.add(Embedding(100,80,input_length=100))
model.add(Conv1D(10,2,activation='relu'))
model.add(Conv1D(50,2,activation='relu'))
model.add(Conv1D(10,2,activation='relu'))
model.add(Conv1D(40,2,activation='relu'))
model.add(Conv1D(10,2,activation='relu'))
model.add(Conv1D(30,2,activation='relu'))
model.add(Conv1D(10,2,activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(46,activation='softmax'))

#3 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy' , optimizer='adam' , metrics=['acc']  )
model.fit(x_train,y_train,epochs=10000 , batch_size= 100 , validation_split=0.2 , callbacks=[es] )

#4 평가 , 예측
loss = model.evaluate(x_test,y_test)
pre = model.predict(x_test)

y_test = np.argmax(y_test)
pre = np.argmax(pre)

print('loss = ',loss)
print('ACC = ', accuracy_score(y_test,pre) )




