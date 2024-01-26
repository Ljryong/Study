from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train) , (x_test,y_test) = fashion_mnist.load_data()

x_train= x_train/255.
x_test= x_test/255.


train_datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip=True,           # 수평 뒤집기
    vertical_flip=True,             # 수직 뒤집기
    width_shift_range=0.2,          # 가로이동 비율
    height_shift_range=0.2,         # 세로이동 비율
    rotation_range=50,              # 회전 각도 조절
    zoom_range=0.2,                 # 축소, 확대 비율 조절
    shear_range=0.8,                # 찌끄러뜨리거나 , 누르기 
    fill_mode='nearest',
    
)

argumet_size = 40000

randidx = np.random.randint(x_train.shape[0], size = argumet_size)               # 랜덤한 인트값을 뽑는다.
        # np.random.randint(60000,40000)                                        # 60000개 중 40000개를 랜덤으로 뽑는다

print(randidx)      # [ 4709  5920 14810 ... 45827 18883  1793]
print(np.min(randidx), np.max(randidx) )                # 0 59999

x_augummented = x_train[randidx].copy()          # 원데이터에 영향을 미치지 않기 위해서 .copy() 를 쓴다

y_augummented = y_train[randidx].copy()

print(x_augummented)
print(x_augummented.shape)          # (40000, 28, 28)
print(y_augummented)
print(y_augummented.shape)          # (40000,)


x_augummented = x_augummented.reshape(40000,28,28,1)
            # = x_augummented.reshape(-1,28,28,1)
            # = x_augummented.reshape(x_augummented[0],x_augummented[1],x_augummented[2],1)
            
print(x_augummented)       
print(x_augummented.shape)        # (40000, 28, 28, 1)
            
            
x_augummented = train_datagen.flow(
    x_augummented, y_augummented ,
    batch_size=argumet_size,
    shuffle = False
).next()[0]                    # .next 뒤에 [0] 을 쓰면 x 값만 나온다.


print(x_augummented[0])         # [0]은 x [1]은 y

print(x_train.shape)                # (60000, 28, 28)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)


print(x_train.shape,x_augummented.shape)


x_train = np.concatenate((x_train,x_augummented))              #concatenate: 사슬처럼 엮다
y_train = np.concatenate((y_train,y_augummented))
print(x_train.shape,y_train.shape)          # (100000, 28, 28, 1) (100000,)

