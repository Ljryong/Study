import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential , Model
from keras.layers import Input , Dense, concatenate , Concatenate
from keras.callbacks import EarlyStopping


#1 데이터
x1_datasets = np.array([range(100),range(301,401)]).T             # 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201),range(411,511),range(150,250)]).T      # 원유 , 환율 , 금시세
                                                        # .T  =  열과 행을 바꿔주는 것 
print(x1_datasets.shape,x2_datasets.shape)              # (100, 2) (100, 3)

x3_datasets = np.array([range(100),range(301,401),range(77,177),range(33,133)]).T


y = np.array(range(3001,3101))      # 비트코인 종가

# 전체 데이터의 개수는 맞춰줘야한다.

x1_train, x1_test ,x2_train,x2_test,x3_train,x3_test, y_train , y_test  =train_test_split(x1_datasets,x2_datasets ,x3_datasets, y,test_size=0.3,random_state=12)

# x2_train, x2_test , y_train , y_test = train_test_split(x2_datasets,y,test_size=0.3,random_state=12)

# print(x1_train.shape,x2_train.shape,y_train.shape)              # (70, 2) (70, 3) (70,) 2개만이 아니라 여러개도 한번에 자를 수 있다. (2개이상)

es = EarlyStopping(monitor='val_loss' , mode = 'min' , patience= 500 , restore_best_weights=True )

#2-1 모델
input1 = Input(shape=(2,))
d1 = Dense(10,activation='relu',name='bit1')(input1)
d2 = Dense(10,activation='relu',name='bit2')(d1)
d3 = Dense(10,activation='relu',name='bit3')(d2)
output1 = Dense(10,activation='relu',name='bit4')(d3)

# model1=Model(inputs = input1 , outputs = output1)
# model1.summary()



#2-2
input11 = Input(shape=(3,))
d11 = Dense(100,activation='relu',name='bit11')(input11)
d12 = Dense(100,activation='relu',name='bit12')(d11)
d13 = Dense(100,activation='relu',name='bit13')(d12)
output11 = Dense(5,activation='relu',name='bit14')(d13)

# model2=Model(inputs = input11 , outputs = output11)
# model2.summary()

#2-3
input21 = Input(shape=(4,))
d21 = Dense(10,activation='relu')(input21)
d22 = Dense(10,activation='relu')(d21)
d23 = Dense(10,activation='relu')(d22)
output21 = Dense(10,activation='relu')(d23)


#2-4   concatenate
#   아웃풋 나온 2개를 합쳐줘야되는데 히든레이어 없이 합쳐주면 아웃풋의 개수가 많은것에 쏠리기 때문에 섞기 위해서 히든레이어 층을 만들어줘야된다.
merge1 = concatenate([output1,output11,output21], name = 'mg1' )                                         # merge : 합치다 , 병합
merge2 = Dense(7,name='mg2')(merge1)
merge3 = Dense(11,name = 'mg3')(merge2)
last_output = Dense(1,name='last')(merge3)

model = Model(inputs = [input1,input11,input21] , outputs = [last_output] )
model.summary()


#3 컴파일 , 훈련
model.compile(loss = 'mse' , optimizer='adam' )
model.fit([x1_train,x2_train,x3_train],y_train, epochs= 10000 , batch_size= 100 , validation_split= 0.2 ,callbacks=[es])

#4 평가 , 예측
loss = model.evaluate([x1_test,x2_test,x3_test] , y_test)
predict = model.predict([x1_test,x2_test,x3_test])

print('loss',loss)
print('predict',predict)
# predict가 30개인 이유는 train_test_split으로 잘랐기 때문이다.0.3비율


'''
loss 6.159146437312302e-08
predict [[3018.0005]
 [3042.    ]
 [3093.    ]
 [3015.    ]
 [3069.    ]
 [3032.    ]
 [3089.9995]
 [3016.0002]
 [3022.    ]
 [3060.9998]
 [3013.    ]
 [3009.    ]
 [3040.    ]
 [3010.0005]
 [3008.    ]
 [3071.    ]
 [3059.    ]
 [3025.    ]
 [3087.0002]
 [3017.    ]
 [3083.9998]
 [3055.9995]
 [3026.9998]
 [3054.9998]
 [3020.0005]
 [3058.    ]
 [3047.    ]
 [3024.0002]
 [3037.0005]
 [3092.    ]]
'''