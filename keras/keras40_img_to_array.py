import sys
import tensorflow as tf
print("텐서플로 버전" , tf.__version__)     # 텐서플로 버전 2.9.0
print("파이썬 버전" , sys.version)          # 파이썬 버전 3.9.18 (main, Sep 11 2023, 14:09:26) [MSC v.1916 64 bit (AMD64)]
import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator         # 이미지를 불러워서 수치화 하는 놈
from tensorflow.keras.preprocessing.image import load_img      # 이미지를 불러오는 애
from tensorflow.keras.preprocessing.image import img_to_array  # 이미지를 수치화 하는 애
# from keras.utils import load_img , img_to_array

path = 'c:/_data/image/catdog/Train/Cat//1.jpg'
img = load_img(path ,
                 target_size = (150,150),
                 )

print(img)
# <PIL.Image.Image image mode=RGB size=150x150 at 0x2119F1A7B80>
print(type(img))    # <class 'PIL.Image.Image'>

plt.imshow(img)
plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)    # (150, 150, 3)         # (281, 300, 3) = 원 데이터 크기 target size를 주석처리하면 원본의 크기가 뽑힌다.
print(type(arr))    # <class 'numpy.ndarray'>

##### 차원증강 #####
img = np.expand_dims(arr,axis=0)                # 차원을 늘려라
print(img.shape)                                # axis = 0 일 때 , (1, 150, 150, 3)
                                                # axis = 1 일 때 , (150, 1, 150, 3)
                                                # axis = 2 일 때 , (150, 150, 1, 3)

