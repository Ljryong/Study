from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense , LSTM , Embedding , Conv1D , Flatten
from sklearn.metrics import accuracy_score




# 결과 긍정 부정인지 찾기
#1 데이터
docs = ['너무 재미있다.' , '참 최고에요.' , '참 잘만든 영화에요.' , '추천하고 싶은 영화입니다.' , '정룡이가 한 번 더 보고 싶어요' ,
        '글쎄' , '별로에요' , '생각보다 지루해요' , '연기가 어색해요' , '재미없어요' , 
        '너무 재미없다' ,'참 재밋네요' , '상헌이 바보' , '반장 정말 잘생겼다', '욱이 도 잔다.',
        '나는 잘생겼다' , '반장 싫다' ]

labels = np.array([1,1,1,1,1,
                   0,0,0,0,0,
                   0,1,0,1,0,
                   1,0])
x_predict = ['나는 정룡이가 정말 싫다. 재미없다 너무 정말']

token = Tokenizer()
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)


print(type(x))          # <class 'list'>
# x = np.array(x)       차원이 달라서 안됨
 
token.fit_on_texts(docs)
x_predict = token.texts_to_sequences(x_predict)



'''

from keras.utils import pad_sequences
pad_x = pad_sequences(x, padding='pre' ,                                # padding 디폴트가 앞(pre)으로 되어있다. post로 하면 뒤로 나옴
                      maxlen=5 ,                                        # maxlen=10 은 최대 길이를 10까지 늘린다.
                      truncating='post')                                # maxlen 을 넘어가는 데이터를 자르는것 post는 뒤에 pre 는 앞에
print(pad_x)
# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  0  1  5  6]
#  [ 0  0  7  8  9]
#  [10 11 12 13 14]
#  [ 0  0  0  0 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0 17 18]
#  [ 0  0  0 19 20]
#  [ 0  0  0  0 21]
#  [ 0  0  0  2 22]
#  [ 0  0  0  1 23]
#  [ 0  0  0 24 25]
#  [ 0  0  0 26 27]
#  [ 0  0 28 29 30]]
print(pad_x.shape)              # (15, 5)


pad_x = pad_x.reshape(-1,6,1)

print(pad_x.shape)          # (15, 5, 1)
#2 모델구성
model = Sequential()
model.add(Embedding(input_dim = 31 , output_dim= 10 , input_length=5 )) # , input_length= 5 이 없어도 돌아감 굳이 쓸 필요 X
model.add(LSTM(10))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()



#3 컴파일, 훈련
model.compile(loss = 'binary_crossentropy' , optimizer='adam', metrics=['acc'])
model.fit(pad_x , labels, epochs=1000 , batch_size= 10  )

#4 평가, 예측
loss = model.evaluate(pad_x,labels)
pre = model.predict(x_predict)
pre = np.round(pre)

print('loss = ',loss)
print('acc = ',accuracy_score(labels,pre))


# Epoch 1000/1000
# 2/2 [==============================] - 0s 10ms/step - loss: 5.6325e-05 - acc: 1.0000
# 1/1 [==============================] - 0s 155ms/step - loss: 5.5922e-05 - acc: 1.0000
# 1/1 [==============================] - 0s 129ms/step
# loss =  [5.592157685896382e-05, 1.0]
# acc =  1.0




'''