from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2 모델구성
model = Sequential()
model.add(Dense(3 , input_dim = 1))             # 첫번째 model.add 말고는 input을 넣지 않아도 위에 input 기록이 남아있어서 쓸 수 있다.
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(1))


#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=10000)

#4 평가,예측
loss = model.evaluate(x,y)
print(loss)
result = model.predict([4])
print(result)



# 5.5868176751516785e-09
# 1/1 [==============================] - 0s 55ms/step
# [[4.000094]]