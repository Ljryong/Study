from keras.preprocessing.text import Tokenizer

text = '나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token =Tokenizer()
token.fit_on_texts([text])              # 2개 이상도 가능

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8} 많이 나온걸 번저 정리하고 그 다음 앞에있는걸 정리해줌
#                                                                                           나온 순서가 동일하면 앞에 있는게 먼저

print(token.word_counts)
# OrderedDict([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)]) 몇번 나왔는지 세어줌

x = token.texts_to_sequences([text])
print(x)
# [[4, 2, 2, 3, 3, 5, 6, 7, 1, 1, 1, 8]] 회귀모델에 그냥 넣게되면 1 이랑 8이 8배가 차이나게 되는 수가 되기때문에 OneHot으로 바꿔줘야된다.

from keras.utils import to_categorical
import numpy as np
##################################

x = to_categorical(x)           # to_categorical 은 0번부터 나오는데 text는 1부터 나와서 처리를 해줘야 한다.
# [[[0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]

x =np.array(x)
x = np.delete(x,0,axis=1)                   # np.delete(삭제할 데이터(배열), 삭제할 위치 , axis = (0행 ,1열 ))
print(x)

#1. to_categorical에서 첫번째 0을 빼는 법
#2 사이킷런의 OneHotEncoder를 사용
#3 판다스의 getdummies 를 사용
##################################

# from sklearn.preprocessing import OneHotEncoder
# x =np.array(x)                      # 중요 리스트를 array 형식으로 바꿔주는것
# print(x.shape)
# x = x.reshape(-1,1)                 # reshape를 먼저 하고 fit을 해야한다.
# encoder = OneHotEncoder()
# encoder.fit(x)

# x = encoder.transform(x).toarray()
# print(x.shape)                  # (12, 8)
# [[0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1.]]

##################################

import pandas as pd
x =np.array(x)
x = x.reshape(-1)               # 벡터의 형태로 만들어줘야 사용할 수 있음
x = pd.get_dummies(x).astype(int)               # .astype(int) 를 사용해주면 False, True 로 나오는게 아니라 숫자로 나온다(다른것들과 동일한 모양으로)
                                                # 혹은 판다스의 버전이 낮아지면 0,1 로 나온다.
                                                
# print(x)
#         1      2      3      4      5      6      7      8
# 0   False  False  False   True  False  False  False  False
# 1   False   True  False  False  False  False  False  False
# 2   False   True  False  False  False  False  False  False
# 3   False  False   True  False  False  False  False  False
# 4   False  False   True  False  False  False  False  False
# 5   False  False  False  False   True  False  False  False
# 6   False  False  False  False  False   True  False  False
# 7   False  False  False  False  False  False   True  False
# 8    True  False  False  False  False  False  False  False
# 9    True  False  False  False  False  False  False  False
# 10   True  False  False  False  False  False  False  False
##################################
