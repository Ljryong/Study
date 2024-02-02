from keras.models import Sequential
from keras.layers import Dense , Dropout , LeakyReLU , BatchNormalization
from keras.callbacks import EarlyStopping , ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder , LabelEncoder , MinMaxScaler
import datetime
from imblearn.over_sampling import SMOTE
import time


#1
path = 'c:/_data/dacon/dechul//'
train_csv = pd.read_csv(path+ 'train.csv' , index_col= 0)
test_csv = pd.read_csv(path+ 'test.csv' , index_col= 0)
submission_csv = pd.read_csv(path+ 'sample_submission.csv')

encoder = LabelEncoder()
encoder.fit(train_csv['주택소유상태'])
train_csv['주택소유상태'] = encoder.transform(train_csv['주택소유상태'])
encoder.fit(train_csv['대출목적'])
train_csv['대출목적'] = encoder.transform(train_csv['대출목적'])
encoder.fit(train_csv['대출기간'])
train_csv['대출기간'] = encoder.transform(train_csv['대출기간'])
encoder.fit(train_csv['근로기간'])
train_csv['근로기간'] = encoder.transform(train_csv['근로기간'])


encoder.fit(test_csv['주택소유상태'])
test_csv['주택소유상태'] = encoder.transform(test_csv['주택소유상태'])
encoder.fit(test_csv['대출목적'])
test_csv['대출목적'] = encoder.transform(test_csv['대출목적'])
encoder.fit(test_csv['대출기간'])
test_csv['대출기간'] = encoder.transform(test_csv['대출기간'])
encoder.fit(test_csv['근로기간'])
test_csv['근로기간'] = encoder.transform(test_csv['근로기간'])

train_csv = train_csv[train_csv['주택소유상태'] != 'ANY']           # 주택소유상태에서 any가 들어간것만 삭제 // != : 부정 // 행을 삭제함


encoder.fit(train_csv['대출등급'])
train_csv['대출등급'] = encoder.transform(train_csv['대출등급'])

date = datetime.datetime.now()
date = date.strftime('%m%d-%H%M')
path = 'c:/_data/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path , 'k28_11_', date , '_', filename ])


x = train_csv.drop(['대출등급'],axis = 1 )
y = train_csv['대출등급']


y = y.values.reshape(-1,1)       # (96294, 1)




x_train ,x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3, random_state= 96 , shuffle=True , stratify=y)    # 0 1502 96 27
es = EarlyStopping(monitor='val_loss', mode='min' , patience= 400 , restore_best_weights=True , verbose= 1 )




# print(y_train.shape)            # (67405, 7) // print(y_train.shape) = output 값 구하는 법


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)        # test_csv도 같이 학습 시켜줘야 값이 나온다. 안해주면 소용이 없다.


#2

from keras.optimizers import Adam
def build_model(hp):
    model = Sequential()
        # Dense Layer에 unit수 선택
    # 정수형 타입 32부터 512까지 32배수 범위 내에서 탐색
        # activation 은 relu 사용
    model.add(Dense(units=hp.Int('units',
                                        min_value=32,
                                        max_value=512,
                                        step=32),
                           activation='relu'))

    model.add(Dense(10, activation='softmax'))
    model.compile(
        optimizer=Adam(
        # 학습률은 자주 쓰이는 0.01, 0.001, 0.0001 3개의 값 중 탐색
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


from keras_tuner.tuners import Hyperband
tuner = Hyperband(
        build_model, # HyperModel
        objective ='val_accuracy', #  최적화할 하이퍼모델
        max_epochs =20, # 각 모델별 학습 회수
        factor = 3,    # 한 번에 훈련할 모델 수 결정 변수
        directory ='my_dir', # 사용된 parameter 저장할 폴더
        project_name ='helloworld',
        ) # 사용된 parameter 저장할 폴더

# 작성한 Hypermodel 출력
tuner.search_space_summary()
# tuner 학습
tuner.search(x_train, y_train,
             epochs=10,validation_split = 0.2)
# 최고의 모델을 출력
model = tuner.get_best_models(num_models=2)[0]
# 혹은 결과 출력
tuner.results_summary()




#3
from keras.callbacks import EarlyStopping ,ModelCheckpoint
mcp = ModelCheckpoint(monitor='val_loss', mode='min' , verbose=1, save_best_only=True , filepath=  filepath   )


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train, epochs = 10000000 , batch_size= 300 , validation_split=0.2 , callbacks = [es,mcp] , verbose= 2 )


#4
loss = model.evaluate(x_test,y_test)

y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict,axis=1)
submit =  np.argmax(y_submit,axis=1)


# sparse_categorical_crossentropy 을 쓰면 y_test 에는 argmax를 사용할 필요가 없다


y_submit = encoder.inverse_transform(submit)       # inverse_transform 처리하거나, 뽑을라면 argmax처리를 해줘야한다.
submission_csv['대출등급'] = y_submit







submission_csv.to_csv(path+'submission_0202.csv', index = False)


print('y_submit = ', y_submit)

print('loss = ',loss)
print("f1 = ",f1_score(y_test,y_predict,average='macro'))


# y_submit =  ['B' 'B' 'B' ... 'B' 'C' 'B']
# loss =  [222.89456176757812, 0.43975216150283813]
# f1 =  0.30516756362369907

# Epoch 357: early stopping
# 753/753 [==============================] - 1s 732us/step - loss: 48.2262 - acc: 0.4793
# 2007/2007 [==============================] - 1s 695us/step
# 753/753 [==============================] - 1s 683us/step
# y_submit =  ['C' 'A' 'A' ... 'C' 'C' 'A']
# loss =  [48.226158142089844, 0.47931379079818726]
# f1 =  0.37544917638611197

# 2007/2007 [==============================] - 2s 812us/step
# 753/753 [==============================] - 1s 840us/step
# y_submit =  ['B' 'A' 'A' ... 'D' 'B' 'A']
# loss =  [11.950640678405762, 0.4693860709667206]
# f1 =  0.4320287071351898


# MinMaxScaler
# Epoch 2274: early stopping
# 903/903 [==============================] - 1s 576us/step - loss: 0.4005 - acc: 0.8647
# 2007/2007 [==============================] - 1s 553us/step
# 903/903 [==============================] - 1s 571us/step
# y_submit =  ['B' 'B' 'A' ... 'D' 'C' 'A']
# loss =  [0.400475412607193, 0.8646543622016907]
# f1 =  0.8219496330591821

# StandardScaler
# Epoch 2172: early stopping
# 903/903 [==============================] - 1s 593us/step - loss: 0.3664 - acc: 0.8739
# 2007/2007 [==============================] - 1s 540us/step
# 903/903 [==============================] - 1s 560us/step
# y_submit =  ['B' 'A' 'A' ... 'D' 'C' 'A']
# loss =  [0.3664281964302063, 0.8739312291145325]
# f1 =  0.8449073685924191

# MaxAbsScaler
# Epoch 2230: early stopping
# 903/903 [==============================] - 1s 592us/step - loss: 0.4184 - acc: 0.8541
# 2007/2007 [==============================] - 1s 557us/step
# 903/903 [==============================] - 1s 560us/step
# y_submit =  ['B' 'B' 'A' ... 'D' 'C' 'A']
# loss =  [0.41843244433403015, 0.8540620803833008]
# f1 =  0.7846639869016788

# RobustScaler
# Epoch 2237: early stopping
# 903/903 [==============================] - 1s 571us/step - loss: 0.3488 - acc: 0.8790
# 2007/2007 [==============================] - 1s 548us/step
# 903/903 [==============================] - 0s 537us/step
# y_submit =  ['B' 'A' 'A' ... 'D' 'C' 'A']
# loss =  [0.34875673055648804, 0.8790196776390076]
# f1 =  0.8497940978155077




# #2
# model = Sequential()
# model.add(Dense(102 ,input_shape= (13,)))
# model.add(Dense(15,activation= 'swish'))
# model.add(Dense(132,activation= 'swish'))
# model.add(Dense(13, activation= 'swish'))
# model.add(Dense(64,activation= 'swish'))
# model.add(Dense(7,activation='softmax'))
# Epoch 3597: early stopping
# 903/903 [==============================] - 1s 876us/step - loss: 0.2848 - acc: 0.9255
# 2007/2007 [==============================] - 1s 505us/step
# 903/903 [==============================] - 0s 545us/step
# y_submit =  ['B' 'B' 'A' ... 'D' 'C' 'A']
# loss =  [0.2847670018672943, 0.9255425930023193]
# f1 =  0.9094408160608632
# = 0.92



