# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np          # 수치 계산이 빠름
import pandas as pd         # 수치 말고 다른 각종 계산들이 좋고 빠름
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

#1. 데이터

path = "c:/_data/dacon/ddarung//"

train_csv = pd.read_csv(path + "train.csv",index_col = 0) # index_col = 0 , 필요없는 열을 지울 때 사용한다 , index_col = 0 은 0번은 index야 라는 뜻

test_csv = pd.read_csv(path + "test.csv", index_col = 0)          # [715 rows x 10 columns] = [715,10] -- index_col = 0 사용하기 전 결과 값
print(test_csv)

submission_csv = pd.read_csv(path + "submission.csv", )   # 서브미션의 index_col을 사용하면 안됨 , 결과 틀에서 벗어날 수 있어서 index_col 을 사용하면 안됨

print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())         # describe는 함수이다 , 함수 뒤에는 괄호가 붙는다. 수치 값을 넣어야 사용할 수 있기 때문에 괄호를 붙여야 된다.


train_csv = train_csv.interpolate()
test_csv = test_csv.interpolate()
              
######### x 와 y 를 분리 ############

print(train_csv.isna().sum())

def outliers(data_out):
    quartile_1 , q2 , quartile_3 = np.percentile(data_out,[25,50,75])       # 25,50,75 퍼센트로 나눔
    print('1사 분위 :' , quartile_1 )
    print('q2 :' , q2 )
    print('3사 분위 :' , quartile_3 )
    iqr = quartile_3 - quartile_1                       
    # 이상치는 보통의 값을 벗어난 것인데 이상치는 엄청 크거나 엄청 작거나 둘중 하나이다
    # 이런걸 방지하기위해서 상위25%과 하쉬25%를 버리고 나머지 50%를 가져온다.
    # 가운데 데이터들은 보통 정상적인 데이터라고 판단(아닐수도 잇음) 
    print('iqr :' , iqr)
    lower_bound = quartile_1 - (iqr * 1.5)              
    # 1.5가 아니여도 되는데 통상 1.5가 제일 좋음
    # 로우 = 4 - (6 * 1.5) = 4 - 9 = -5 여기까지의 데이터를 이상치가 아니라고 판단한다
    upper_bound = quartile_3 + (iqr * 1.5)
    # 하이 = 10 + (6 * 1.5) = 10 + 9 = 19 여기까지의 데이터를 이상치가 아니라고 판단
    print(lower_bound)  # -69.25
    print(upper_bound)  # 118.35
    return np.where((data_out>upper_bound) | (data_out<lower_bound))        # | python 함수에서 or 이랑 같은 뜻이다
    # 2가지 조건중에 한개라도 만족하는걸 빼냄 19큰거 -5보다 작은걸 빼내라
    # 뽑으면 위치값 0 , 12 의 값이 이상치라고 나옴

outliers_loc = outliers(train_csv)
print('이상치의 위치 :' , outliers_loc)
print(len(outliers_loc[0]))     # 1510
# 1사 분위 : 1.1
# q2 : 17.7
# 3사 분위 : 48.0
# iqr : 46.9
# 이상치의 위치 : (array([   0,    1,    2, ..., 1456, 1457, 1458], dtype=int64), array([5, 5, 5, ..., 5, 5, 5], dtype=int64))
# 1510

# 이상치 제거
train_csv = train_csv[train_csv['hour_bef_pm10']<=120]
import matplotlib.pyplot as plt
# plt.boxplot(x)
# plt.show()

print(train_csv.isna().sum())
x = train_csv.drop(['count'],axis = 1)                # 'count'를 drop 해주세요 axis =1 에서 (count 행(axis = 1)을 drop 해주세요) // 원본을 건드리는 것이 아니라 이 함수만 해당
print(x)
y = train_csv['count']                                # count 만 가져오겠다
print(y)



x_train, x_test, y_train , y_test = train_test_split(x,y,test_size = 0.3, random_state= 846 ) #45

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor

model = RandomForestRegressor()
# model = RandomForestClassifier()

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
print(x.shape)


'''

# 쓰기 전 좋은 성적
# n_components =  8 result 0.6994900687800266

# 쓰고난 후 성적
hour_bef_pm10 컬럼의 이상치가 너무 많아 잘라버렸다
Name: count, Length: 1405, dtype: float64
model.score 0.7708003855985753
(1405, 9)

'''
