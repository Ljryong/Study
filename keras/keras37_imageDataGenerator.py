import numpy as np
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1/255.,
                                   horizontal_flip=True,     # 수평 뒤집기
                                   vertical_flip=True,       # 수직 뒤집기
                                   width_shift_range=0.1,    # 평행이동 / 수평이동 0.1이면 10%까지 랜덤으로 이동한다(1%도 되고 10%까지 다 됨)
                                   height_shift_range=0.1,   # 평행이동 / 수직이동 0.1이면 10%까지 랜덤으로 이동한다
                                   rotation_range=5,         # 정해진 각도만큼 이미지를 회전
                                   zoom_range=1.2,           # 축소 또는 확대
                                   shear_range=0.7,          # 좌표하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환
                                   fill_mode='nearest'       # 너의 빈자리를 가장 비슷한 책으로 채워
                                                             # 0로 나타내 주는것도 잇으니 찾아보기
                                    )

test_datagen = ImageDataGenerator(rescale=1./255)      # test 데이터는 train에서 훈련한 것과 비교하는 실제 데이터로 해야되기 때문에 rescale만 쓴다.

path_train = 'c:/study/image/brain/train//'
path_test = 'c:/study/image/brain/test//'


xy_train = train_datagen.flow_from_directory(                                   # 그림들을 가져와서 수치화해주는 것 (이터레이터형태)
                                             path_train,                        # 데이터를 가져오는 경로
                                             target_size = (200,200) ,          # 내가 정한 수치까지 그림들의 사이즈를 줄이거나 늘린다// 
                                             batch_size =  160,                 # 160 이상의 수를 쓰면 x의 통 데이터(160)로 들어간다
                                             class_mode='binary',               # 2중 분류를 나타내는 것, 다중 분류 일 때에는 categorical을 사용한다. 
                                             shuffle=True                       # 그림을 섞는 것 
                                             )       


# Found 160 images belonging to 2 classes.


# from sklearn.datasets import load_diabetes
# datasets = load_diabetes()
# print(datasets)

xy_test = test_datagen.flow_from_directory(                            
                                             path_test,                 
                                             target_size = (200,200) ,   
                                             batch_size =  160, 
                                             class_mode='binary',
                                             shuffle=False              # test는 섞는것이 아니다 // 건드리면 데이터 조작이다.
                                             )       

# Found 120 images belonging to 2 classes.

print(xy_train)
 # <keras.preprocessing.image.DirectoryIterator object at 0x000001CC26C83520>

print(xy_train.next())              # 1번째 값만 보여준다
print(xy_train[0])
# print(xy_train[16])               # 에러 : 전체데이터/batch_size = 10 이라서 160/10은 16개인데 
                                    #       [16]는 17번째의 값을 빼라고 해서 에러가 나는것이다.

print(xy_train[0][0])       # 첫번째의 배치 x
# xy_train[0][0] 첫번째 대괄호는 batch_size를 나타내는 것이다, 총 160개를 batch 10 으로 했다면 15까지 나오는거고 통 데이터를 x,y로 나누기 위해 batch를 160으로 주고
# 0번째 배치 즉 160개의 데이터를 가르키는 것이다. 두번째 대괄호는 
print(xy_train[0][1])       # 두번째의 배치 y
print(xy_train[0][0].shape)             # (10, 200, 200, 3) 흑백은 칼라다 o //칼라는 흑백이다 X

print(type(xy_train))             # <class 'keras.preprocessing.image.DirectoryIterator'>       Iterator형태 = 2개의 데이터로 나눌수 있는 형태로 가지고 있다.
# print(type(xy_train[0]))        # <class 'tuple'>
# print(type(xy_train[0][0]))     # <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))     # <class 'numpy.ndarray'>




