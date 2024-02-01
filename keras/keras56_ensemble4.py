import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential , Model
from keras.layers import Input , Dense, concatenate , Concatenate
from keras.callbacks import EarlyStopping


#1 데이터
x1_datasets = np.array([range(100),range(301,401)]).T             # 삼성전자 종가, 하이닉스 종가

y1 = np.array(range(3001,3101))      # 비트코인 종가
y2 = np.array(range(13001,13101))    # 이더리움 종가

# 전체 데이터의 개수는 맞춰줘야한다.

x_train, x_test , y1_train , y1_test ,y2_train,y2_test = train_test_split(x1_datasets, y1,y2,test_size=0.3,random_state=12)

# x2_train, x2_test , y_train , y_test = train_test_split(x2_datasets,y,test_size=0.3,random_state=12)

# print(x1_train.shape,x2_train.shape,y_train.shape)              # (70, 2) (70, 3) (70,) 2개만이 아니라 여러개도 한번에 자를 수 있다. (2개이상)

es = EarlyStopping(monitor='val_loss' , mode = 'min' , patience=300 , restore_best_weights=True )

#2-1 모델
input1 = Input(shape=(2,))
d1 = Dense(10,activation='relu',name='bit1')(input1)
d2 = Dense(10,activation='relu',name='bit2')(d1)
d3 = Dense(10,activation='relu',name='bit3')(d2)
output1 = Dense(1,name='bit4')(d3)


output2 = Dense(1)(d3)

model=Model(inputs = input1 , outputs = [output1,output2])
# model1.summary()


#3 컴파일 , 훈련
model.compile(loss = 'mse' , optimizer='adam' )
model.fit(x_train,[y1_train,y2_train], epochs= 10000 , batch_size= 100 , validation_split= 0.2 ,callbacks=[es])

#4 평가 , 예측
loss = model.evaluate(x_test, [y1_test,y2_test])
predict = model.predict(x_test)

print('loss',loss)
print('predict',predict)
# predict가 30개인 이유는 train_test_split으로 잘랐기 때문이다.0.3비율


'''
loss [3.5881996609532507e-06, 5.364418029785156e-07, 3.051757857974735e-06]
predict [array([[3018.0012],
       [3042.    ],
       [3092.9993],
       [3015.001 ],
       [3069.    ],
       [3032.001 ],
       [3089.9993],
       [3016.0012],
       [3022.001 ],
       [3061.    ],
       [3013.0007],
       [3009.0007],
       [3040.0002],
       [3010.0015],
       [3008.0012],
       [3070.9998],
       [3059.0002],
       [3025.0007],
       [3086.9998],
       [3017.0012],
       [3083.9998],
       [3056.0002],
       [3027.0005],
       [3055.    ],
       [3020.0005],
       [3057.9998],
       [3047.0005],
       [3024.0005],
       [3037.0007],
       [3091.9993]], dtype=float32), array([[13017.999],
       [13041.998],
       [13093.001],
       [13014.999],
       [13069.001],
       [13031.999],
       [13090.   ],
       [13015.999],
       [13021.999],
       [13061.   ],
       [13012.997],
       [13008.997],
       [13039.999],
       [13009.998],
       [13007.997],
       [13071.001],
       [13059.   ],
       [13024.997],
       [13087.002],
       [13017.001],
       [13084.002],
       [13056.001],
       [13026.998],
       [13054.999],
       [13019.997],
       [13057.998],
       [13047.001],
       [13023.997],
       [13036.998],
       [13092.001]], dtype=float32)]
'''