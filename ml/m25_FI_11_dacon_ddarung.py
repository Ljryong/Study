# https://dacon.io/competitions/open/235576/mysubmission

import numpy as np          # 수치 계산이 빠름
import pandas as pd         # 수치 말고 다른 각종 계산들이 좋고 빠름
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.svm import LinearSVR



#1. 데이터

path = "c:/_data/dacon/ddarung//"
# print(path + "aaa_csv") = c:/_data/dacon/ddarung/aaa_csv


train_csv = pd.read_csv(path + "train.csv",index_col = 0) # index_col = 0 , 필요없는 열을 지울 때 사용한다 , index_col = 0 은 0번은 index야 라는 뜻
# \\ 는 2개씩 해야한다 , 하지만 파일 경로일 때는 \ 1개여도 가능                                                                    
# \ \\ / // 다 된다, 섞여도 가능하지만 가독성에 있어서 한개로 하는게 좋다


print(train_csv)     # [1459 rows x 11 columns] = [1459,11] -- index_col = 0 사용하기 전 결과 값

test_csv = pd.read_csv(path + "test.csv", index_col = 0)          # [715 rows x 10 columns] = [715,10] -- index_col = 0 사용하기 전 결과 값
print(test_csv)

submission_csv = pd.read_csv(path + "submission.csv", )   # 서브미션의 index_col을 사용하면 안됨 , 결과 틀에서 벗어날 수 있어서 index_col 을 사용하면 안됨
print(submission_csv)

print(train_csv.shape)      # (1459, 10)
print(test_csv.shape)         # (715, 9)
print(submission_csv.shape)   # (715, 2)            test 랑 submission 2개가 id가 중복된다.

print(train_csv.columns)        
# #Index(['id', 'hour', 'hour_bef_temperature', 'hour_bef_precipitation',
# 'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
# 'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
print(train_csv.info())
#      Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64
# dtypes: float64(9), int64(1)
print(test_csv.info())
#      Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    715 non-null    int64
#  1   hour_bef_temperature    714 non-null    float64
#  2   hour_bef_precipitation  714 non-null    float64
#  3   hour_bef_windspeed      714 non-null    float64
#  4   hour_bef_humidity       714 non-null    float64
#  5   hour_bef_visibility     714 non-null    float64
#  6   hour_bef_ozone          680 non-null    float64
#  7   hour_bef_pm10           678 non-null    float64
#  8   hour_bef_pm2.5          679 non-null    float64
# dtypes: float64(8), int64(1)

print(train_csv.describe())         # describe는 함수이다 , 함수 뒤에는 괄호가 붙는다. 수치 값을 넣어야 사용할 수 있기 때문에 괄호를 붙여야 된다.

######### 결측치 처리 ###########
# 1.제거
'''
print(train_csv.isnull().sum())             # isnull 이랑 isna 똑같다
# print(train_csv.isna().sum())
train_csv = train_csv.dropna()              # 결측치가 1행에 1개라도 있으면 행이 전부 삭제된다
# print(train_csv.info())                   # 결측치 확인 방법
print(train_csv.shape)                      # (1328, 10)      행무시, 열우선
                                            # test data는 결측치를 제거하는 것을 넣으면 안된다. test data는 0이나 mean 값을 넣어줘야 한다.
'''

# 결측치 평균값으로 바꾸는 법
# train_csv = train_csv.fillna(train_csv.mean())  

test_csv = test_csv.fillna(test_csv.mean())                    # 717 non-null     



##################### 결측치를 0으로 바꾸는 법#######################

train_csv = train_csv.fillna(0)

                                          

######### x 와 y 를 분리 ############
x = train_csv.drop(['count'],axis = 1)                # 'count'를 drop 해주세요 axis =1 에서 (count 행(axis = 1)을 drop 해주세요) // 원본을 건드리는 것이 아니라 이 함수만 해당
print(x)
y = train_csv['count']                                # count 만 가져오겠다
print(y)

''' 25퍼 미만 열 삭제 '''
# columns = datasets.feature_names
columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
fi_str = "0.03546066 0.03446177 0.43028001 0.49979756"
 
''' str에서 숫자로 변환하는 구간 '''
fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
print(fi_float)
fi_list = pd.Series(fi_float)

''' 25퍼 미만 인덱스 구하기 '''
low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print('low_idx_list',low_idx_list)

''' 25퍼 미만 제거하기 '''
low_col_list = [x.columns[index] for index in low_idx_list]
# 이건 혹여 중복되는 값들이 많아 25퍼이상으로 넘어갈시 25퍼로 자르기
if len(low_col_list) > len(x.columns) * 0.25:   
    low_col_list = low_col_list[:int(len(x.columns)*0.25)]
print('low_col_list',low_col_list)
x.drop(low_col_list,axis=1,inplace=True)
print("after x.shape",x.shape)


x_train, x_test, y_train , y_test = train_test_split(x,y,test_size = 0.3, random_state= 846 ) #45

print(x_train.shape, x_test.shape)                    # (1021, 9) (438, 9)---->(929, 9) (399, 9) == 결측치를 제거 했을 때
print(y_train.shape, y_test.shape)                    # (1021,) (438,) ------>(929,) (399,)  == 결측치를 제거 했을 때


#2 모델구성
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
models = [DecisionTreeRegressor(random_state = 777), RandomForestRegressor(random_state = 777) , 
          GradientBoostingRegressor(random_state = 777),XGBRegressor()]

############## 훈련 반복 for 문 ###################a
for model in models :
    model.fit(x_train,y_train)
    result = model.score(x_test,y_test)
    print(type(model).__name__,':',model.feature_importances_ ,result)
   # y_predict = model.predict(x_test)
    print(type(model).__name__,'result',result)


# 로스는 :  2943.575927734375 = 54.2

# 로스는 :  3048.900634765625
# 14/14 [==============================] - 0s 0s/step
# R2 =  0.555676685166897


# 로스는 :  2841.63037109375
# 14/14 [==============================] - 0s 1ms/step
# R2 =  0.5748732131682246

# 로스는 :  2928.133544921875
# 14/14 [==============================] - 0s 1ms/step
# R2 =  0.5972511865433818


# 로스는 :  2953.03564453125
# 14/14 [==============================] - 0s 1ms/step
# R2 =  0.5938260415210248

# 로스는 :  2711.775390625
# 14/14 [==============================] - 0s 0s/step
# R2 =  0.5738032160590182

# 로스는 :  2816.08935546875
# 14/14 [==============================] - 0s 0s/step
# R2 =  0.6120126733503751

# 로스는 :  2663.163818359375
# 14/14 [==============================] - 0s 0s/step
# R2 =  0.6348010541990886

# 로스는 :  2676.828369140625
# 14/14 [==============================] - 0s 1ms/step
# R2 =  0.6329272259885607

# 로스는 :  2652.173583984375
# 14/14 [==============================] - 0s 216us/step
# R2 =  0.6363080952424065



# 로스는 :  0.06708519905577148
# R2 =  0.06708519905577148



# LinearSVR score  0.5262892442851996
# LinearSVR predict  0.5262892442851996
# LinearRegression score  0.6072759286909343
# LinearRegression predict  0.6072759286909343
# RandomForestRegressor score  0.7950294977012764
# RandomForestRegressor predict  0.7950294977012764
# DecisionTreeRegressor score  0.5414605596521274
# DecisionTreeRegressor predict  0.5414605596521274
# KNeighborsRegressor score  0.4068639187657872
# KNeighborsRegressor predict  0.4068639187657872



# DecisionTreeRegressor : [0.60530913 0.17484584 0.02998343 0.02865885 0.03903972 0.02541971
#  0.0331521  0.04101667 0.02257456] 0.5540727966071619
# DecisionTreeRegressor result 0.5540727966071619
# RandomForestClassifier : [0.10321384 0.1446559  0.0051876  0.12968747 0.13017558 0.1168987
#  0.12663902 0.12601325 0.11752863] 0.01598173515981735
# RandomForestClassifier result 0.01598173515981735
# GradientBoostingRegressor : [0.65855721 0.2100336  0.02317936 0.01301977 0.01804376 0.02554294
#  0.02621237 0.02018394 0.00522705] 0.7754493511906716
# GradientBoostingRegressor result 0.7754493511906716
# XGBRegressor : [0.29117462 0.07511689 0.48993376 0.01196224 0.02641661 0.02371508
#  0.03317604 0.0322391  0.01626559] 0.798534725966388
# XGBRegressor result 0.798534725966388
