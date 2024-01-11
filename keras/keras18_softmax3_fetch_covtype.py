from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape,y.shape)      # (581012, 54) (581012,)
print(pd.value_counts(y))   # 2    283301 , 1    211840 , 3     35754 , 7     20510 , 6     17367 , 5      9493 , 4      2747   (n,7)

from keras.utils import to_categorical          # 얘가 안됨
one_hot = to_categorical(y)
print(one_hot)

# one_hot = pd.get_dummies(y)

# from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(-1,1)
# ohe = OneHotEncoder()
# ohe.fit(y)
# one_hot = ohe.transform(y).toarray()

# print(one_hot)



x_train , x_test , y_train , y_test = train_test_split(x,one_hot,test_size=0.3 , random_state=0 ,shuffle=True, stratify=y )

es= EarlyStopping(monitor='val_loss' , mode = 'min', verbose= 1 ,patience=10, restore_best_weights=True )


print(datasets.DESCR)


'''
#2
model = Sequential()
model.add(Dense(2048,input_dim = 54))
model.add(Dense(1024))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(7,activation='softmax'))

#3
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])  # 오류 뜨는거 해결을 위해 sparse_categorical_crossentropy를 넣어봄
model.fit(x_train, y_train, epochs = 1 , batch_size=1000000, verbose=1 , validation_split=0.2, callbacks=[es] )



#4
result = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test , axis =1)
y_predict = np.argmax(y_predict, axis = 1)

print(y_test.shape)
print(y_predict.shape)

acc = accuracy_score(y_test,y_predict)

print('acc = ' , acc)




'''



