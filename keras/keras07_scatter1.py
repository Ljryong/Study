import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split                # 사이킷런 돌리기 위해서 가져오는 정보


#1 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=0)         
# 입력된 정보중에서 랜덤으로 test_size = 0.3만큼을 빼는 것 , test_size 0.3 은 train_size 0.7 과 같다. train_size 의 디폴트 값은 0.75 이다.
# train_size 와 test_size 합계가 1이 넘으면 돌아가지 않는다. 합이 1이 되지않을시에는 손실이 일어난다. train이 0.5고 test가 0.4면 손실이 0.1 일어난다.
# sklearn.model_selection 여기서 train_test_split 이 정보를 가져올 때 x_train,x_test,y_train,y_test 이 순서대로 가져와서 변환되면 안된다. 결과 값이 달라진다.
# shuffle = False 는 숫자가 섞이지 않고 뒤에서 부터 자름 shuffle 의 디폴트 값은 ture 이다.
# 


print(x_train)
print(y_train)
print(x_test)
print(y_test)

# 검색 : train과 test를 섞어서 7:3 으로 자를 수 있는 방법 찾기
# 힌트 : 사이킷런

#2 모델구성
model = Sequential()
model.add(Dense(5,input_dim = 1))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1000,batch_size = 1)

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
result = model.predict(x)
print("loss = ", loss)
print("result = ", result)

# 4/4 [==============================] - 0s 0s/step - loss: 0.1275
# 1/1 [==============================] - 0s 78ms/step - loss: 0.4102
# 1/1 [==============================] - 0s 66ms/step
# loss =  0.41015252470970154
# result =  [[10.732424]]

import matplotlib.pyplot as plt

plt.scatter(x,y)                        # scatter = 흩뿌리다
plt.plot(x,result,color='red')
plt.show()