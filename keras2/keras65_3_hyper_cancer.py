# CNN 으로 만든다.
# early stop 도 파라미터로 적용 시키기
# MCP 적용
# [실습] : 맹그러

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential , Model
from keras.layers import Dense , Conv2D , Flatten , MaxPooling2D , Dropout , Input
from keras.callbacks import EarlyStopping , ModelCheckpoint
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#1 데이터
datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , stratify=y , random_state=123 , shuffle= True )

def build_model(drop=0.5 , optimizer='adam' , activation ='relu' , node1=32 ,node2 = 32 ,node3 = 16 , lr = 0.001 ) : 
    # 밑에 명시하지 않았으면 자동으로 위에 명시한 값들이 들어감
    inputs =Input(shape=(30,) ,name='inputs' )
    x = Dense( node1 , activation=activation , name = 'hidden1' )(inputs)
    x = Dropout(drop)(x)
    x = Dense( node2 , activation=activation , name = 'hidden2' )(x)
    x = Dropout(drop)(x)
    x = Dense( node3 , activation=activation , name = 'hidden3' )(x)
    outputs = Dense(1,activation = 'sigmoid' , name='outputs' )(x)

    model = Model(inputs = inputs ,outputs = outputs )

    model.compile(optimizer = optimizer, metrics=['mse'] , loss = 'binary_crossentropy' )
    
    return model

def create_hyperparameter():
    batchs = [32,64,100]
    optimizers = ['adam' , 'rmsprop' , 'adadelta' ]
    dropouts = [0.2,0.3,]
    activations = ['relu', 'elu','swish' ]
    node1 = [64,32,16]
    node2 = [64,32,16]
    node3 = [64,32,16]
    
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    es = EarlyStopping(monitor='loss',patience=5,mode='auto',restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='loss',patience=3,mode='auto',factor=0.7)
    callback = [es,rlr]

    return {'batch_size' : batchs , 'optimizer' : optimizers , 'drop' : dropouts , 'activation' : activations ,
            'node1':node1, 'node2':node2 , 'node3':node3 ,'callbacks':callback }

hyperparameter = create_hyperparameter()
print(hyperparameter)

# keras 와 엮기
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier , KerasRegressor
# keras의 모델을 rapping 해줄 모델

keras_model = KerasRegressor(build_fn=build_model,  # 위에 만든 keras 모델을 sklearn의 형태로 rapping 시켜줌
                              verbose = 1
                              )

# model = RandomizedSearchCV(build_model , hyperparameter , cv=2 , n_iter=1 , n_jobs=-1 , verbose= 1  )
# 이렇게 하면 Randomizedsearch에 내가 지정해준 파라마터가 없어서 에러가 뜸 , 모델도 머신러닝이 아니라서 못받아들임
# 이걸 해결하기 위해서 keras 모델을 sklearn 의 ML 처럼 만들어줌

model = RandomizedSearchCV(keras_model , hyperparameter , cv = 2 , n_iter= 3 , n_jobs= 20 , verbose= 1  )

import time
start = time.time()
model.fit(x_train , y_train , epochs = 3  )
end = time.time()

print('걸린 시간 : ', round(end - start,2) )
print('model.best_params_ : ' , model.best_params_ )
print('model.best_estimator_ : ' , model.best_estimator_ )
print('model.best_score_ : ' , model.best_score_)               # train으로 판단
print('model.score : ',model.score(x_test , y_test))            # test 로 판단 이게 좀 더 정확함

from sklearn.metrics import accuracy_score , r2_score
y_predict = model.predict(x_test)
print('accuracy : ' , accuracy_score(y_test,y_predict) )

# 걸린 시간 :  548.06
# model.best_params_ :  {'optimizer': 'adam', 'node3': (3, 3), 'node2': (2, 2), 'node1': (4, 4), 'drop': 0.3, 'batch_size': 64, 'activation': 'swish'}
# model.best_estimator_ :  <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001FC06168400>
# model.best_score_ :  0.9932666718959808
# 157/157 [==============================] - 0s 2ms/step - loss: 0.0490 - acc: 0.9879 
# model.score :  0.9879000186920166
# 313/313 [==============================] - 0s 764us/step
# accuracy :  0.9879