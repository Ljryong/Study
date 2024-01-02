import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]]
             )      #칼럼을 2개로 만들면 괄호를 한개 더 쳐야한다. 칼럼=열

y = np.array([1,2,3,4,5,6,7,8,9,10])            #(1,2,3,4,5,6,7,8,9,10) = (10,)

print(x.shape)         #(2, 10)  =  (행, 열)          #shape : 모양  (2,10) 2개의 행과 10개의 열의 모양
print(y.shape)         #(10,)


