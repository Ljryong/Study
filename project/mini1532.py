import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Dense , Flatten
from keras.models import Sequential

hub_url = "https://www.kaggle.com/models/google/movinet/frameworks/TensorFlow2/variations/a5-stream-kinetics-600-classification/versions/2"

encoder = hub.KerasLayer(hub_url, trainable= False , input_shape = (None,320,320,3) )

model = Sequential()
model.add(encoder)
model.add(Dense(10,activation='softmax'))
model.summary()