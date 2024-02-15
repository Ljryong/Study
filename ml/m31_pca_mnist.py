from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

(x_train, _ ) , (x_test, _ ) = mnist.load_data()
# 받고 싶지 않은게 있을 때 _를 넣어주면 됨(에러가 뜨지 않음)
print(x_train.shape,x_test.shape)           # (60000, 28, 28) (10000, 28, 28)

# x = np.append(x_train,x_test,axis=0)
x = np.concatenate([x_train , x_test],axis=0)       # 지금 사용한 append 와 concatenate는 같다
print(x.shape)      # (70000, 28, 28)

# 실습
# pca 를 통해 0.95 이상인 n_components는 몇개

x = x.reshape(-1,x.shape[1]*x.shape[2])

# scaler = StandardScaler()
# x = scaler.fit_transform(x)

pca = PCA(n_components=x.shape[1])
pca.fit_transform(x)
evr = pca.explained_variance_ratio_

pca_cumsum = np.cumsum(evr)
print( 'n_components:',784 ,'변화율', pca_cumsum)

print(np.argmax(pca_cumsum >= 0.95) + 1)    # 154  //
print(np.argmax(pca_cumsum >= 0.99) + 1)    # 331  //
print(np.argmax(pca_cumsum >= 0.999) + 1)   # 486  //
print(np.argmax(pca_cumsum >= 1.0) + 1)     # 713  //  784-713 = 71


# count1 = sum(1 for num in pca_cumsum if num > 0.95 )
# count2 = sum(1 for num in pca_cumsum if num > 0.99 )
# count3 = sum(1 for num in pca_cumsum if num > 0.999 )
# count = sum(1 for num in pca_cumsum if num == 1 )
# # count4 = sum(1 for num in pca_cumsum if num  )
# print(count1)
# print(count2)
# print(count3)
# print(count)

# # 453
# 241
# 102





