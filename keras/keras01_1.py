import tensorflow as tf # tensorflow를 땡겨오고, tf라고 줄여서 쓴다.
print(tf.__version__)   # 2.15.0
from tensorflow.keras.models import Sequential  # tensorflow에서 sequential을 땡겨온다
from tensorflow.keras.layers import Dense       # tensorflow에서 Dense를 땡겨온다
import numpy as np  # np로 바꾸는건 속도가 빨라서 바꾸는 것 


#1. 데이터 (데이터 정제 or 데이터 전처리)
x = np.array([1,2,3])   # np데이터 123을 준비한 것
y = np.array([1,2,3])   

#2. 모델구성            #y=wx+b  x = input , y = output
model = Sequential()                    # Sequential = 순차적으로 나열하겠다
model.add(Dense(1, input_dim=1))        # y = 1 = output = 한덩어리 ,  x = dim = 하나의 차원 = 한덩어리  ,  dense = 밀집도


#3. 컴파일, 훈련           # 반복하면서 loss 를 줄여가며 최적의 weight를 구한다
model.compile(loss='mse',optimizer='adam')   # loss = mse 를 쓸거고 optimizer = adam 을 쓸거다 complime = 이걸 이걸로 해줘 loss 를 mse 로 해줘 optimizer를 adam 으로 해줘
# loss = 실제값과 예측값의 차이(제일 낮은 값이 좋고 양수 밖에 나올 수 없다, 절대값) / mse = 제곱하는 방식으로 거리를 잰다
model.fit(x,y,epochs=9000)                  # adam = loss 값을 건드려주는 아이, 일단은 그냥 좋다는 것만 기억 , fit = 훈련한다 , epochs = 몇번 돌릴지 횟수  , 최적의 weight 생성
                                # 과접합에 걸릴 수 있어서 훈련량을 조절해야한다

#4. 평가, 예측                              # 최적의 weight값으로 평가랑 예측을 진행한다
loss = model.evaluate(x,y)      # evaluate = 평가
print("로스 : ", loss)
result = model.predict([4])     # predict = 예측  //    결과 값은 y의 다음 값을 구하는 것이다.
print("4의 예측값은", result)

# , = 분리하는것 . = 이어주는 것



