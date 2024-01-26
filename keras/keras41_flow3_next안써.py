from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train) , (x_test,y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,           # 수평 뒤집기
    vertical_flip=True,             # 수직 뒤집기
    width_shift_range=0.2,          # 가로이동 비율
    height_shift_range=0.2,         # 세로이동 비율
    rotation_range=50,              # 회전 각도 조절
    zoom_range=0.2,                 # 축소, 확대 비율 조절
    shear_range=0.8,                # 찌끄러뜨리거나 , 누르기 
    fill_mode='nearest',
    
)

argumet_size = 100

print(x_train[0].shape)  # (28, 28)
# plt.imshow(x_train[0])        밑에꺼만 주석하고 이라인에 있는것을 주석하지 않으면 마지막에 다른걸 뽑을 때 섞여서 나온다.
# plt.show()

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), argumet_size ).reshape(-1,28,28,1),
    # np.tile(x,y) : x는 반복할 배열 / y는 반복할 수 
    # 위의 타일의 뜻은  x_train[0]을 reshape 해서 1차원 배열로 만들고 argument_size(100) 만큼 붙인다.

    np.zeros(argumet_size),
    batch_size=argumet_size,
    shuffle=False,
)           # .next를 뺴면 괄호가 하나 더 있는 상태로 튜플 형태로 넣는것
            # .next를 쓰면 for 문에 들어가서 반복할때 자동으로 넘어간다.
             
print(x_data)       
# print(x_data.shape)   # 튜플은 shape가 되지 않음 / flow에서 나올때 튜플형태로 반환됨
print(x_data[0][0].shape)   # (100, 28, 28, 1) / next를 쓰지않아서 1개씩 밀림
print(x_data[0][1].shape)   # (100,)

print(np.unique(x_data[0][1],return_counts=True))      # (array([0.]), array([100], dtype=int64))
# np.zeros(argumet_size) 을 써서 y가 전부 0인 것이다.
print(x_data[0][0][0].shape)        # 마지막 괄호는 

plt.figure(figsize=(7,7))

for i in range(49):             # 49 는 몇번 반복할지를 나타내는 것 // 
                                # len(xy_train)으로 쓰는이유는 배치로 자른것을 전부 실행시키기 위해서 쓰는것이다. 
                                # len(xy_train)은 batch를 자른 것으로 전체데이터를 돌리면 몇번이 실행됬는지를 숫자로 나타내줌 
    plt.subplot(7,7,i+1)
    plt.axis('off')         # 축 끄기
    plt.imshow(x_data[0][0][i],cmap='gray')
plt.show()    







