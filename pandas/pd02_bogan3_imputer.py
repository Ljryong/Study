import numpy as np
import pandas as pd

data = pd.DataFrame([[2,np.nan,6,8,10],
                     [2,4,np.nan,8,np.nan],
                     [2,4,6,8,10],
                     [np.nan,4,np.nan,8,np.nan],
                     ])
# print(data)
data = data.transpose()
data.columns = ['x1','x2','x3','x4']
# print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer , KNNImputer , IterativeImputer
from sklearn.impute import IterativeImputer

imputer = SimpleImputer()

data2 = imputer.fit_transform(data)             # imputer 의 default는 결측치가 mean(평균)으로 나옴
# print(data2)



imputer = SimpleImputer(strategy='mean')    # 평균
data3 = imputer.fit_transform(data)         
# print(data3)

imputer = SimpleImputer(strategy='median')  # 중위
data4 = imputer.fit_transform(data)         
# print(data4)

imputer = SimpleImputer(strategy='most_frequent')  # 가장 자주 나오는 애
data4 = imputer.fit_transform(data)         
# print(data4)

imputer = SimpleImputer(strategy='constant')  # 상수 / 고정값은 0
data5 = imputer.fit_transform(data)         
# print(data5)

imputer = SimpleImputer(strategy='constant' , fill_value=777)  # 상수로 채우는데 fill_value 로 숫자를 원하는것으로 정해줄 수 있다.
data6 = imputer.fit_transform(data)         
# print(data6)

imputer = KNNImputer()
data7 = imputer.fit_transform(data)
# print(data7)
# KNN = 범위 안에 가장 많은 애를 따라감 = 근처에 많은 놈으로 따라감

imputer = IterativeImputer()                # Intterpolate랑 비슷한 놈이다.
                                            # 선형회귀 알고리즘
data8 = imputer.fit_transform(data)         
# print(data8)

# Iterative랑 mice 둘 다 선형 회귀이지만 mice 다중 선형 회귀 방식이다.

print(np.__version__)               # 1.26.3

from impyute.imputation.cs import mice
mc = mice(data)
print(mc)

