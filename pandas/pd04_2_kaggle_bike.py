from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR


#1 데이터

path = 'c:/_data/kaggle/bike//'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

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
    return np.where((data_out>upper_bound) | (data_out<lower_bound))        # | python 함수에서 or 이랑 같은 뜻이다
    # 2가지 조건중에 한개라도 만족하는걸 빼냄 19큰거 -5보다 작은걸 빼내라
    # 뽑으면 위치값 0 , 12 의 값이 이상치라고 나옴
outliers_loc = outliers(train_csv)
print('이상치의 위치 :' , outliers_loc)

train_csv = train_csv[train_csv['humidity']>=40]
train_csv = train_csv[train_csv['windspeed']<=31]
train_csv = train_csv[train_csv['casual']<=50]
train_csv = train_csv[train_csv['registered']<=250]
train_csv = train_csv[train_csv['count']<=300]

#  6   humidity    10886 non-null  int64
#  7   windspeed   10886 non-null  float64
#  8   casual      10886 non-null  int64
#  9   registered  10886 non-null  int64
#  10  count       10886 non-null  int64

import matplotlib.pyplot as plt
# plt.boxplot(train_csv)
# plt.show()
# print(train_csv.info())

x = train_csv.drop(['count'], axis= 1 )        # [6493 rows x 8 columns] // drop을 줄 때 '를 따로 따로 줘야된다.
y = train_csv['count']


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3 , random_state= 6974 ) #7

es = EarlyStopping(monitor = 'val_loss' , mode = 'min', patience = 10 , verbose= 1 ,restore_best_weights=True )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

###################
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2 모델구성
from sklearn.ensemble import RandomForestRegressor


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=6)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
#2
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor

model = RandomForestRegressor()
# model = RandomForestClassifier()

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
print(x.shape)

evr = pca.explained_variance_ratio_

evr_cumsum = np.cumsum(evr)   
print(evr_cumsum)


# 사용전
# 이상치의 위치 : (array([   13,    14,    15, ..., 10883, 10884, 10884], dtype=int64), array([10, 10, 10, ..., 10,  9, 10], dtype=int64))
# model.score 0.9735808629648908


# 사용후
# 이상치 이상들을 잘라냈을경우 성능이 떨어짐
# model.score 0.9476030748307218
# (6484, 10)
# [0.25607201 0.43508654 0.56378697 0.67848541 0.77652489 0.85820847]