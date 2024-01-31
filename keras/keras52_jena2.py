# 5일분(720행)을 훈련시켜서 하루(144행) 뒤를 예측한다. 


from keras.models import Sequential
from keras.layers import Dense , Dropout , SimpleRNN , LSTM ,GRU
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
import time 

#1 데이터
start = time.time()
path = 'c:/_data/kaggle/jena//'

xy = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)

# print(xy.shape)             # (420551, 14)


x  = xy
y = xy['T (degC)']

timestep = 720

def split(xy,timestep,col) : 
    x=[]
    y=[]
    for i in range(len(xy) - timestep-144) :
        a,b = xy[i : (i+timestep)] , xy.iloc[i+timestep+144][col]           # iloc는 pandas가 : 을 잘 인식하지 못해서 인식하기 위해 쓰는것이다.
                                                                # pandas 인걸 확인하는 법은 읽어올때 pandas로 읽어서 그런것이고 이것 말고도 type으로 찍어서 확인이 가능하다.
        x.append(a)
        y.append(b)
    return np.array(x) , np.array(y)
# iloc = index 를 제외한 수치(값)들만을 뽑아내준다. 
x , y = split(xy,timestep,'T (degC)')

end = time.time()

# [-21.31 -20.97 -20.48 ...  -3.16  -4.23  -4.82]


print(x,y)

print(x.shape,y.shape)       # (419687, 720, 14) (419687,)s


x_train , x_test , y_train, y_test = train_test_split(x,y, test_size = 0.3 , random_state = 0  , shuffle = True )
es = EarlyStopping(monitor='val_loss'  , mode = 'min' , patience= 50 , restore_best_weights=True , verbose= 1  )



#2 모델구성
model = Sequential()
model.add(GRU(1,input_shape = (720,14),activation='relu'))


#3 컴파일, 훈련
model.compile(loss = 'mse' , optimizer='adam' , metrics=['mse'])
model.fit(x_train,y_train, epochs= 1 , batch_size= 1 , validation_split=0.2 , callbacks=[es] )


#4 평가, 예측
loss = model.evaluate(x_test,y_test)
predict = model.predict(x_test)


print('loss = ' ,loss)
print('결과 = ' ,predict)
print('시간' , end - start)



# LSTM
# loss =  [0.44846034049987793, 0.44846034049987793]
# 결과 =  [[18.965517 ]
#  [-1.9998345]
#  [ 0.7080312]
#  ...
#  [16.786592 ]
#  [-6.1850452]
#  [ 5.322441 ]]


# GRU
# loss =  [0.048605289310216904, 0.048605289310216904]
# 결과 =  [[18.904936 ]
#  [-1.8903892]
#  [ 1.4161297]
#  ...
#  [16.741688 ]
#  [-5.7747936]
#  [ 5.4525313]]




# loss =  [11.742328643798828, 11.742328643798828]
# 결과 =  [[21.188663]
#  [17.17575 ]
#  [15.77075 ]
#  ...
#  [ 9.421377]
#  [18.269936]
#  [12.87629 ]]
# 시간 17.797977924346924




