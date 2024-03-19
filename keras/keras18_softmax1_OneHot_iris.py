import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

#1
datasets = load_iris()      # target에 0 ,1 ,2 만 봐도 벌써 3개니까 softmax를 사용
# print(datasets)         # (n,4)
# print(datasets.DESCR)       # 라벨의 개수 = 클래스의 개수
# print(datasets.feature_names)   # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets.data
y = datasets.target

# print(x.shape,y.shape)  # shape는 무조건 확인하기       (150, 4) (150,)

# # 회귀모델인지 분류모델인지 확인하는 법
# print(y)                # 회귀모델인지 분류모델인지 꼭 확인하야 됨
# print(np.unique(y,return_counts=True))      # (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
# print(pd.value_counts(y))                   # 라벨의 개수가 차이가 많이나면 한쪽으로 쏠려서 과적합이 일어난다.
#                                             # 1:8:1 로 되어있으면 1들을 증폭시켜 8:8:8로 만들어줘야한다.(6:4)정도까진 괜찮음

########## keras로 onehot 쓰는법############

from keras.utils import to_categorical

one_hot = to_categorical(y)     # y를 one_hot 으로 지정한것
print(one_hot)
print(one_hot.shape)
# 0과 1로 나옴 
# one_hot = to_categorical() 을 사용하면 1열이 늘어난다.

########## pandas로 onehot ################

# one_hot = pd.get_dummies(y)
# print(one_hot)
# print(one_hot.shape)
# True , False로 나옴 , 문제는 없지만 문제가 생긴다면 바꿔주면 됨

########## sklearn으로 onehot (1) 조금 복잡함 #############
# from sklearn.preprocessing import OneHotEncoder
# y_ohe = y.reshape(-1,1)     # (150,1) // 벡터를 행렬로 바꾸는 행동
# y_ohe = y.reshape(150,1)    # (150,1)

# print(y_ohe)
# ohe = OneHotEncoder(sparse = False).fit(y_ohe)          
# y_ohe = ohe.transform(y_ohe)

# print(y_ohe)
# print(y_ohe.shape)

# ########## sklearn으로 onehot (2) #############
from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(-1,1)
# ohe = OneHotEncoder(sparse = False)    # 클래스는 한번씩 정의를 해줘야된다. ex) model = Sequential()
#                                         # OneHotEncoder의 default 값은 True 이다
# ohe.fit(y)                          # 바꿀거라 생각만 하고 맞추는 단계. 아직 바뀌지 않음 , 메모리만 저장 되어있는 상태
# y_ohe = ohe.transform(y_ohe)        # fit으로 생각했던걸 실행하는 단계

# y_ohe1 = ohe.fit_transform(y)        # 위에 2개를 1개로 줄인 것 // 똑같은 것이다.        fit + transform = fit_transform
#                                     # 줄인게 무조건 좋은건 아니다 지금은 1개지만 나중에 2개이상 나올 때 사용이 불편하다

# y_ohe1 = ohe.fit_transform(y).toarray()       # .toarray() 는 sparse = True 일때만 가능 



# print(y_ohe1)
# print(y_ohe1.shape)



'''
#################### 중요 #####################
y_ohe = y.reshape(-1,1)     # (150,1) // 벡터를 행렬로 바꾸는 행동
y_ohe = y.reshape(150,1)    # (150,1)

y = y.reshape(50,3)         # 가능 // 데이터가 바뀌었는지가 중요 순서랑 내용 , 개수가 바뀌지 않았어여 가능 // reshape에선 (2,3)=(3,2)=(6,)
print(y)
y = y.reshape(50,4)         # 불가능
print(y)
'''


x_train , x_test , y_train , y_test = train_test_split(x,one_hot,test_size= 0.2 , random_state= 2 , shuffle=True, stratify = y , )       # stratify = y 는 분류모델에서만 쓴다
# stratify = y 라벨을 골고루 잡아줌 ex) y = 0이 20개 1이 40개로 0.5 를 주면 0은 10개 1은 20개가 빠지게된다
# 분류모델에서 라벨중 특정 값만 많이 들어갈 수 있어서 고루 분배되게 해야된다.
# stratify = 층을 형성시키다.
# y 값이 아니라 one_hot이 된 값을 넣어줘야 한다.

print(y_test)
print(np.unique(y_test,return_counts=True))

es = EarlyStopping(monitor='val_loss' , mode = 'min' , verbose= 1 , patience=10 , restore_best_weights=True)



#2 모델구성
model = Sequential()
model.add(Dense(64,input_dim = 4 , activation= 'sigmoid'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(3,activation='softmax'))        # 3개 나오니까 output 이 3 이다. (0,1,2)


#3 컴파일,훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam' , metrics = ['acc'] ) # 값을 뽑으면 metrics에 있는 acc 가 loss 다음으로 나온다.
model.fit(x_train,y_train,epochs = 1000000 , batch_size = 1 , verbose = 1 , validation_split=0.2 , callbacks = [es] )

#4 평가, 예측
result = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print('result , acc :',result)          # 이렇게 하면 값이 2개 나오고

print("loss" , result[0])               # 이렇게 하면 1개 1개 따로 나온다.
print("acc" , result[1])

print(y_predict)


print(y_predict.shape , y_test.shape)       # (30,3) (30,3) // 돌아간다면 0점짜리다.

# y_test = y_test.reshape(90,)          # 이렇게 하면 뒤진다..  //  [0,1,0] = [0.1,0.8,0.1] 은 둘 다 합이 1이라 괜찮은데 
# y_predict = y_predict.reshape(90,)        
# print(y_predict.shape , y_test.shape) # (90,) (90,)       이렇게 바꿀시 0과 0.1 1과 0.8 0과 0.1을 비교하게 되는 행동이다.


y_test = np.argmax(y_test,axis = 1)             # 다중분류모델에서 하는 작업 / [0.1,0.8,0.1] 값 중에 제일 높은놈을 위치값으로 바꾸는 작업
y_predict = np.argmax(y_predict, axis=1)        

# print(y_test)           # [0 2 2 0 1 2 2 2 0 0 1 1 1 2 0 2 1 0 2 0 1 1 0 1 1 2 2 0 1 0]
print(y_predict)                  # (30,)
print(y_test.shape,y_predict)     # (30,)




def ACC(y_test,y_predict) :                 # 순서가 상관이 없다?
    accuracy_score(y_test,y_predict)
acc = accuracy_score(y_test,y_predict)

print("accrucy score : ", acc)

# accrucy score :  1.0


# loss =  [0.078158900141716, 0.9333333373069763]



