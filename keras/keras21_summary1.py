from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score
import numpy as np

#1 데이터

x = np.array([1,2,3])       # (3,1) = (3,)
y = np.array([1,2,3])

# print(x.shape)

#2 모델구성
model = Sequential()
model.add(Dense(5,input_shape = (1,)))      # 차원의 개수를 적는것이다. (열을 세서 열을 적을 수 있고, 스칼라면 차원은 1이다)
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.summary()                             # 요약




