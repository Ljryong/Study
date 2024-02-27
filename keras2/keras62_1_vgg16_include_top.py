import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(777)
np.random.seed(777)
print(tf.__version__)

from keras.applications import VGG16

# model = VGG16()           # default
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________
model = VGG16( # weights='imagenet',            # default
              include_top=False,                # True가 default // fasle를 해야 input을 바꿀 수 있다           
              input_shape=(32,32,3),            # (224,224,3)
              )

# 통상 Dense를 fully connected layer 라 한다

#1 include_top=False가 되면 fully connected layer 가 사라짐
#2 input_shape 바꾸고 싶은대로 바꿀 수 있음 

# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________
model.summary()







