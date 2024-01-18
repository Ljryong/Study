import numpy as np
from keras.models import Sequential , Model
from keras.layers import Dense , Input , Dropout

# 끝선을 맞추지 않으면 오류가 나서 처음 띄어쓰기를 조심해야한다.

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],[1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]])      #칼럼을 2개로 만들면 괄호를 한개 더 쳐야한다. 칼럼=열    (2행 10열)= 2개의 덩어리 안에 열개씩 있다.
y = np.array([1,2,3,4,5,6,7,8,9,10])            #(1,2,3,4,5,6,7,8,9,10) = (10,) = 스칼라가 10개짜리 한 묶음

print(x.shape)         #(2, 10)  =  (행, 열)          #shape : 모양  (2,10) 2개의 행과 10개의 열의 모양
print(y.shape)         #(10,)                         #tensor 이상으로 20문제 만들어서 제출

x = x.T         # x = x.T 로 하면 행렬 전치가 이루어진다. (2,10)에서 (10,2)로 바뀐다 // x.T랑 같은걸로는 x.transpose()가 있다.
print(x.shape)  #(10,2)
#[[1,1],[2,1.1],[3,1.3], ... [10,1.3]]

#2 모델구성 (순차적 모델)
# model = Sequential()
# model.add(Dense(10,input_shape = (2,)))           # 열, 컬럼, 속성, 특성, 차원 = 2 // (행무시,열우선) <= 외우기  (행의 갯수, 열의 갯수)
# model.add(Dense(9))
# model.add(Dense(8,activation = 'relu'))
# model.add(Dense(7))
# model.add(Dense(1))
# model.summary()



#2  모델구성 (함수형)
input1 = Input(shape=(2,))      # 함수형 모델은 어떤모델인지 마지막에 정의를 해준다. / 함수형에는 shape는 가능하지만 dim은 올 수 없다.
dense1 = Dense(10)(input1)      # Dense() 뒤에 오는 괄호가 input 역할을 한다.
dense2 = Dense(9)(dense1)       # Dense(9) = output / dense1 = input 을 넣는다.
dropout = Dropout(0.2)(dense2)  # 함수에서 dropout 쓰는 법
dense3 = Dense(8,activation='relu')(dropout)
dense4 = Dense(7)(dense3)
output1= Dense(1)(dense4)
model = Model(inputs= input1 , outputs = output1 )      # inputs , outputs s를 붙여야된다.

model.summary()                 # sequential 모델은 input 레이어를 보여주지 않지만 함수형은 보여준다.


#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=1000)

#4 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10,1.3]])        
# x와 스칼라 or 벡터를 맞춰줘야 되기 때문에 괄호가 2개이다. 모르겠으면 #1의 x 값과 똑같이 맞추면 된다. 
# (10,1.3)은 (2,) 벡터라서 구하는 값이 아니고 [[10,1.3]]으로 해야 (1,2)로 행렬로 해야한다.

print("[10,1.3]의 예측 값 : " , results)
print("loss의 값 : ",loss)




