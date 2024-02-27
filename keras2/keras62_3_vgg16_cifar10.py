# 성능비교 , 시간체크

import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense , Flatten
from keras.datasets import cifar10
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)
start = time.time()
from keras.applications import VGG16
from keras.callbacks import EarlyStopping

(x_train,y_train) , (x_test,y_test) = cifar10.load_data()

es = EarlyStopping(patience= 10 , monitor='val_loss' , mode='min' , restore_best_weights=True )

vgg16 = VGG16(weights='imagenet' , 
              include_top=False , 
              input_shape=(32,32,3) )

vgg16.trainable = False         # 가중치를 동결(False)한다

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(30))
model.add(Dense(10,activation='softmax'))

model.summary()

#3
model.compile(loss = 'sparse_categorical_crossentropy' , optimizer='adam' , metrics=['acc'])
model.fit(x_train,y_train ,epochs= 1000 , batch_size=1000 , callbacks=[es] , validation_split=0.2 )

#4
loss = model.evaluate(x_test,y_test)

end = time.time()
print('loss',loss[0])
print('acc',loss[1])
print('시간 : ' , end - start)

# Epoch 1301: early stopping
# 313/313 [==============================] - 0s 1ms/step - loss: 0.8316 - acc: 0.7098
# 313/313 [==============================] - 0s 776us/step
# loss 0.831630289554596
# loss_acc 0.7098000049591064
# acc 0.7098

# vgg16.trainable = False
# Epoch 47/1000
# 8/8 [==============================] - 2s 224ms/step - loss: 1.0733 - acc: 0.6281 - val_loss: 1.2041 - val_acc: 0.5907
# 313/313 [==============================] - 1s 4ms/step - loss: 1.1868 - acc: 0.5970
# loss 1.186764121055603
# acc 0.597000002861023
# 시간 :  118.48722124099731

# vgg16.trainable = True
# Epoch 34/1000
# 40/40 [==============================] - 5s 117ms/step - loss: 0.1296 - acc: 0.9570 - val_loss: 1.0946 - val_acc: 0.7659
# 313/313 [==============================] - 1s 4ms/step - loss: 0.8055 - acc: 0.7469
# loss 0.8054966330528259
# acc 0.7469000220298767
# 시간 :  89.86328959465027
