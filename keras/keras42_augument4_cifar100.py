import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar100
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout ,Conv2D , MaxPooling2D ,Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score

(x_train, y_train) , (x_test , y_test) = cifar100.load_data()

x_train = x_train/255.
x_test = x_test/255.

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    rotation_range=50,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
)

# test_datagen = ImageDataGenerator(
#   rescale=1./255
# )

augument_size = 10000

randidx = np.random.randint(x_train.shape[0], size = augument_size)

print(np.min(randidx), np.max(randidx) )    # 2 49999

x_augummented = x_train[randidx].copy()
y_augummented = y_train[randidx].copy()

print(x_augummented.shape)       # (10000, 32, 32, 3)
print(y_augummented.shape)       # (10000, 1)

x_augummented = train_datagen.flow(
    x_augummented,y_augummented,
    batch_size=augument_size,
    shuffle=False
).next()[0]

print(x_train.shape)
print(y_train.shape)


x_train = np.concatenate((x_train,x_augummented)) 
y_train = np.concatenate((y_train,y_augummented))

print('증폭')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


es =  EarlyStopping(monitor='val_loss', mode = 'min' , patience= 100 ,restore_best_weights=True , verbose=1  )


#2 모델구성
model = Sequential()
model.add(Conv2D(150,(2,2),input_shape = (32,32,3),activation='relu', strides=2 , padding='valid'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(Conv2D(14,(2,2),activation='relu',strides=2))
model.add(MaxPooling2D())
model.add(Conv2D(96,(2,2),activation='relu',padding='same'))
model.add(Conv2D(9,(2,2),activation='relu' , padding='same'))
model.add(Flatten())
model.add(Dense(91,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7,activation='relu'))
model.add(Dense(100,activation='softmax'))

#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam' , metrics= ['acc'] )
model.fit(x_train,y_train, epochs= 100000 , batch_size= 1000 , callbacks=[es] , verbose= 1 , validation_split= 0.2  )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

print('loss =',loss)

y_test = np.argmax(y_test,axis=1)
y_predict = np.argmax(y_predict,axis=1)

def ACC(aaa,bbb):
    return accuracy_score(aaa,bbb)
acc = ACC(y_test,y_predict)

print('ACC = ' , acc)


plt.imshow(x_train[15],'gray')
plt.show()




# 증폭
# Epoch 355: early stopping
# 313/313 [==============================] - 0s 1ms/step - loss: 3.2326 - acc: 0.2078
# 313/313 [==============================] - 0s 838us/step
# loss = [3.2325921058654785, 0.2078000009059906]
# ACC =  0.2078