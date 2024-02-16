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

# 결측치 확인
print(data.isna().sum())
print(data.isnull().sum())
# x1    1
# x2    2
# x3    0
# x4    3
# print(data.info())              # info는 항상 확인해야한다

# 결측치 삭제
# print(data.dropna())
# print(data.dropna(axis=1)) # default 는 axis = 0 이다 

# 2-1. 특정값 - 평균
means = data.mean()
# print(means)
# x1    6.500000
# x2    4.666667
# x3    6.000000
# x4    6.000000
data2 = data.fillna(means)
# print(data2)
#      x1        x2    x3   x4
# 0   2.0  2.000000   2.0  6.0
# 1   6.5  4.000000   4.0  4.0
# 2   6.0  4.666667   6.0  6.0
# 3   8.0  8.000000   8.0  8.0
# 4  10.0  4.666667  10.0  6.0

# 2-2. 특정값 - 중위값
med = data.median()
# print(med)
# x1    7.0
# x2    4.0
# x3    6.0
# x4    6.0
data3 = data.fillna(med)
# print(data3)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  6.0
# 1   7.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  6.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  4.0  10.0  6.0

# 2-3. 특정값 - 0 채우기 / 임의의 값 채우기
data4 = data.fillna(0)
# print(data4)
# 0   2.0  2.0   2.0  0.0
# 1   0.0  4.0   4.0  4.0
# 2   6.0  0.0   6.0  0.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  0.0  10.0  0.0

data4_2 = data.fillna(777)
# print(data4_2)

# 2-4. 특정값 - ffill
data5 = data.fillna(method='ffill')
data5 = data.ffill()                # 앞에 값이 없으면 NaN 그대로 나옴
# print(data5)
# 0   2.0  2.0   2.0  NaN
# 1   2.0  4.0   4.0  4.0
# 2   6.0  4.0   6.0  4.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  8.0  10.0  8.0

# 2-5. 특정값 - bfill
data6 = data.fillna(method='bfill')
data6 = data.bfill()                # 뒤에 값이 없으면 NaN 그대로 나옴
# print(data6)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  4.0
# 1   6.0  4.0   4.0  4.0
# 2   6.0  8.0   6.0  8.0
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

'''
주가나 날씨같은 부분(시간을 다루는 부분)에서 ffill, bfill을 많이 사용한다.
'''

################################# 특정 칼럼만 ######################################

means = data['x1'].mean()       # x1의 평균 값만 뽑음
# print(means)        # 6.5

meds = data['x4'].median()
print(meds)     # 4.0


data['x1'] = data['x1'].fillna(means)         # x1에만 결측치에 x1의 평균값을 넣었고 나머지 결측치에는 아무것도 넣지 않음
# print(data)
data['x4'] = data['x4'].fillna(meds)          # x4에만 결측치에 x4의 중위값을 넣었고 나머지 결측치에는 아무것도 넣지 않음
# print(data)
data['x2'] = data['x2'].ffill()               # x2에만 결측치에 x2의 앞의 값을 넣었고 나머지 결측치에는 아무것도 넣지 않음
# print(data)
print(data)







