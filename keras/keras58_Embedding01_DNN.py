from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense 
from sklearn.metrics import accuracy_score


#1 데이터
docs = ['너무 재미있다.' , '참 최고에요.' , '참 잘만든 영화에요.' , '추천하고 싶은 영화입니다.' , '한 번 더 보고 싶어요' ,
        '글쎄' , '별로에요' , '생각보다 지루해요' , '연기가 어색해요' , '재미없어요' , 
        '너무 재미없다' ,'참 재밋네요' , '상헌이 바보' , '반장 잘생겼다', '욱이 도 잔다.'  ]

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)

print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, 
#  '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, 
#  '재미없어요': 21, '재미없다': 22, '재밋네요': 23, '상헌이': 24, '바보': 25, '반장': 26, '잘생겼다': 27, '욱이': 28, '도': 29, '잔다': 30}
# 30개인 이유는 docs 에 쓰인 중복 단어를 제외한 개수

x = token.texts_to_sequences(docs)
print(x)
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14],
# [15], [16], [17, 18], [19, 20], [21]
# , [2, 22], [1, 23], [24, 25], [26, 27], [28, 29, 30]]
# 15개인 이유는 docs의 갯수

print(type(x))          # <class 'list'>
# x = np.array(x)       차원이 달라서 안됨

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




#2 모델구성
model = Sequential()
model.add(Dense(10,input_shape = (5,),activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3 컴파일, 훈련
model.compile(loss= 'binary_crossentropy' , optimizer='adam' , metrics = ['acc'] )
model.fit(pad_x,labels , epochs=1000 ,batch_size= 100)


#4 평가,예측
loss = model.evaluate(pad_x,labels)
y_predict = model.predict(pad_x)

y_predict = np.round(y_predict)

print('loss = ',loss)
print(accuracy_score(labels,y_predict))


# Epoch 1000/1000
# 1/1 [==============================] - 0s 2ms/step - loss: 0.0011 - acc: 1.0000
# 1/1 [==============================] - 0s 82ms/step - loss: 0.0011 - acc: 1.0000
# 1/1 [==============================] - 0s 52ms/step
# loss =  [0.0010772363748401403, 1.0]
# 1.0



