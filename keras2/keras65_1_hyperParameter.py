# 속도 때문에 잘 사용되지 않는 방법
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential , Model
from keras.layers import Dense , Conv2D , Flatten , MaxPooling2D , Dropout , Input

#1 데이터
(x_train , y_train) , ( x_test , y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28*28 ).astype('float32')/255.
x_test = x_test.reshape(10000,28*28 ).astype('float32')/255.

def build_model(drop=0.5 , optimizer='adam' , activation ='relu' , node1=128 ,node2 = 64 ,node3=32 , lr = 0.001 ) : 
    # 밑에 명시하지 않았으면 자동으로 위에 명시한 값들이 들어감
    inputs =Input(shape=(28*28) ,name='inputs' )
    x = Dense(node1 , activation=activation , name = 'hidden1' )(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2 , activation=activation , name = 'hidden2' )(x)
    x = Dropout(drop)(x)
    x = Dense(node3 , activation=activation , name = 'hidden3' )(x)
    x = Dropout(drop)(x)
    outputs = Dense(10,activation = 'softmax' , name='outputs' )(x)
    
    model = Model(inputs = inputs ,outputs = outputs )
    
    model.compile(optimizer = optimizer, metrics=['acc'] , loss = 'sparse_categorical_crossentropy' )
    
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam' , 'rmsprop' , 'adadelta' ]
    dropouts = [0.2,0.3,0.4,0.5]
    activations = ['relu', 'elu', 'selu' , 'linear' ]
    node1 = [128,64,32,16]
    node2 = [128,64,32,16]
    node3 = [128,64,32,16]
    return {'batch_size' : batchs , 'optimizer' : optimizers , 'drop' : dropouts , 'activation' : activations ,
            'node1':node1, 'node2':node2 , 'node3':node3 }

hyperparameter = create_hyperparameter()
print(hyperparameter)

# keras 와 엮기
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier , KerasRegressor
# keras의 모델을 rapping 해줄 모델

keras_model = KerasClassifier(build_fn=build_model,  # 위에 만든 keras 모델을 sklearn의 형태로 wrapping 시켜줌
                              verbose = 1
                              ) 

# model = RandomizedSearchCV(build_model , hyperparameter , cv=2 , n_iter=1 , n_jobs=-1 , verbose= 1  )
# 이렇게 하면 Randomizedsearch에 내가 지정해준 파라마터가 없어서 에러가 뜸 , 모델도 머신러닝이 아니라서 못받아들임
# 이걸 해결하기 위해서 keras 모델을 sklearn 의 ML 처럼 만들어줌

model = RandomizedSearchCV(keras_model , hyperparameter , cv=3 , n_iter=10 , n_jobs=-1 , verbose= 1  )

import time
start = time.time()
model.fit(x_train , y_train , epochs = 3 )
end = time.time()

print('걸린 시간 : ', round(end - start,2) )
print('model.best_params_ : ' , model.best_params_ )
print('model.best_estimator_ : ' , model.best_estimator_ )
print('model.best_score_ : ' , model.best_score_)               # train으로 판단
print('model.score : ',model.score(x_test , y_test))            # test 로 판단 이게 좀 더 정확함

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('accuracy : ' , accuracy_score(y_test,y_predict) )

# 걸린 시간 :  9.25
# model.best_params_ :  {'optimizer': 'rmsprop', 'node3': 64, 'node2': 32, 'node1': 32, 'drop': 0.5, 'batch_size': 500, 'activation': 'elu'}
# model.best_estimator_ :  <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001A1E5436250>
# model.best_score_ :  0.9079999923706055
# 20/20 [==============================] - 0s 839us/step - loss: 0.2547 - acc: 0.9268
# model.score :  0.926800012588501
# 313/313 [==============================] - 0s 415us/step
# accuracy :  0.9268