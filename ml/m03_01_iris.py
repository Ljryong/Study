from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron , LogisticRegression , LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

#1
x,y = load_iris(return_X_y=True)        # sklearn에서 제공하는 데이터만 가능(사실상 실무 나가면 필요 x)


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size= 0.2 , random_state= 2 , shuffle=True, stratify = y )
print(y_test)
print(np.unique(y_test,return_counts=True))

es = EarlyStopping(monitor='val_loss' , mode = 'min' , verbose= 1 , patience=1000000 , restore_best_weights=True)



#2 모델구성
# model = LinearSVC(C=130)            
# model = Perceptron()
# model = LogisticRegression()
# model = RandomForestClassifier()
# model = DecisionTreeClassifier()
model = KNeighborsClassifier()

#3 컴파일,훈련
# model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam' , metrics = ['acc'] ) # 값을 뽑으면 metrics에 있는 acc 가 loss 다음으로 나온다.
# model.fit(x_train,y_train,epochs = 10 , batch_size = 1 , verbose = 1 , validation_split=0.2 , callbacks = [es] )
model.fit(x_train,y_train)      # epoch 도 조절할 수 있지만 대부분 default로 둬도 잘됨 바꿔도 큰 차이가 없는게 대부분이다.

#4 평가, 예측
# result = model.evaluate(x_test,y_test)
result = model.score(x_test,y_test)
# evaluate 가 score 로 바뀜 // score 가 보여주는 지표는 accuracy를 보여준다.(분류모델) // 회귀모델에서의 score는 r2 값을 보여준다.
y_predict = model.predict(x_test)


print("model.score" , result)              

print(y_predict)

acc= accuracy_score(y_predict,y_test)
print(acc)

# print(y_predict.shape , y_test.shape)       # (30,3) (30,3) // 돌아간다면 0점짜리다.

# y_test = y_test.reshape(90,)          # 이렇게 하면 뒤진다..  //  [0,1,0] = [0.1,0.8,0.1] 은 둘 다 합이 1이라 괜찮은데 
# y_predict = y_predict.reshape(90,)        
# print(y_predict.shape , y_test.shape) # (90,) (90,)       이렇게 바꿀시 0과 0.1 1과 0.8 0과 0.1을 비교하게 되는 행동이다.


# y_test = np.argmax(y_test,axis = 1)             # 다중분류모델에서 하는 작업 / [0.1,0.8,0.1] 값 중에 제일 높은놈을 위치값으로 바꾸는 작업
# y_predict = np.argmax(y_predict, axis=1)        

# # print(y_test)           # [0 2 2 0 1 2 2 2 0 0 1 1 1 2 0 2 1 0 2 0 1 1 0 1 1 2 2 0 1 0]
# print(y_predict)                  # (30,)
# print(y_test.shape,y_predict)     # (30,)




# def ACC(y_test,y_predict) :                 # 순서가 상관이 없다?
#     accuracy_score(y_test,y_predict)
# acc = accuracy_score(y_test,y_predict)

# print("accrucy score : ", acc)

# LinearSVC
# accrucy score :  1.0
# loss =  [0.078158900141716, 0.9333333373069763]

# Perceptron
# model.score 0.5666666666666667

# LogisticRegression
# model.score 1.0

# RandomForestClassifier
# model.score 1.0

# DecisionTreeClassifier
# model.score 0.9333333333333333

# KNeighborsClassifier
# model.score 0.9666666666666667


