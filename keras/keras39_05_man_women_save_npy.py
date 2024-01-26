# playlist/men-women-classification
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_datagen = ImageDataGenerator(rescale=1./255)


test_datagen = ImageDataGenerator( rescale=1./255)

path_train = 'C:\\_data\\kaggle\\man_women\\train\\'
path_submit = 'C:\\_data\\kaggle\\man_women\\submit\\'

xy_train = train_datagen.flow_from_directory(
    path_train
    , target_size=(100, 100)
    , batch_size= 2000000
    , class_mode='binary'
    , color_mode='rgb' # default
    , shuffle=True
    # Found 3309 images belonging to 2 classes.
)

test = test_datagen.flow_from_directory(  
    path_submit
    , target_size=(100, 100)
    , batch_size= 2000000
    , class_mode= 'binary'
    , color_mode='rgb' # default
    , shuffle=False)


# print(len(test))


np_path = '../_data/_save_npy/'
np.save(np_path + 'keras39_5_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras39_5_y_train.npy', arr=xy_train[0][1])

# print(xy_train[0][0].shape)
# print(xy_train[0][1].shape)

print('train data ok')

np.save(np_path + 'keras39_5_test.npy' , arr  = test[0][0] )

print('finish')


# Traceback (most recent call last):
#   File "c:\Study\keras\keras39_5_man_women_save_npy.py", line 37, in <module>
#     np.save(np_path + 'keras39_5_test.npy' , arr  = test )
#   File "<__array_function__ internals>", line 200, in save
#   File "C:\Users\AIA\anaconda3\Lib\site-packages\numpy\lib\npyio.py", line 521, in save
#     arr = np.asanyarray(arr)
#           ^^^^^^^^^^^^^^^^^^
#   File "C:\Users\AIA\anaconda3\Lib\site-packages\keras\src\preprocessing\image.py", line 156, in __next__
#     return self.next(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\AIA\anaconda3\Lib\site-packages\keras\src\preprocessing\image.py", line 168, in next
#     return self._get_batches_of_transformed_samples(index_array)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\AIA\anaconda3\Lib\site-packages\keras\src\preprocessing\image.py", line 363, in _get_batches_of_transformed_samples
#     batch_x = np.zeros(
#               ^^^^^^^^^
# numpy.core._exceptions._ArrayMemoryError: Unable to allocate 379. MiB for an array with shape (3309, 100, 100, 3) and data type float32


# 경로 에러



# 남자
# Epoch 36/1000
# 38/38 [==============================] - 1s 14ms/step - loss: 0.0335 - acc: 0.9908 - val_loss: 1.7942 - val_acc: 0.6422
# 32/32 [==============================] - 0s 4ms/step - loss: 0.6021 - acc: 0.6858
# 1/1 [==============================] - 0s 49ms/step
# loss :  0.6020987629890442
# acc :  0.6858006119728088
# [[0.]]