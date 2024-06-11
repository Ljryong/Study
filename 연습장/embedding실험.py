from tensorflow.python.keras.models import Sequential       # == from keras.models import Sequential
# from tensorflow.keras.models import Sequential
# from keras.models import Sequential                         # 3개가 전부 동일하디
from tensorflow.python.keras.layers import Dense , Conv2D  , Embedding     # Conv2D = 2차원이미지 


model = Sequential()
# model.add(Dense(10, input_shape=(3,)))      # 인풋은 (n,3)
model.add(Dense(10,input_dim = 2))         # (2,2)는 자르는 크기 // 1은 흑백 , 3이 컬러 // (10,10,1) == 가로, 세로 10씩의 흑백 사진
model.add(Embedding(10,10))
model.add(Dense(5))
model.add(Dense(1))

model.summary() 