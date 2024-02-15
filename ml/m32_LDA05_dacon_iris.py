from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC


#1
path = "c:/_data/dacon/iris//"

train_csv = pd.read_csv(path + 'train.csv' , index_col = 0)
test_csv = pd.read_csv(path + 'test.csv' , index_col = 0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')


x = train_csv.drop(['species'],axis=1)
y = train_csv['species']

print(pd.value_counts(y))

columns = x.columns
x = pd.DataFrame(x,columns=columns)

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.3, random_state= 200 ,shuffle=True , stratify = y )
# OneHot 을 했으면 y가 아니라 OneHot을 넣어줘야한다.
es = EarlyStopping(monitor='val_loss', mode='min' , verbose= 1 , restore_best_weights=True , patience= 1000  )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.svm import LinearSVR

from sklearn.linear_model import Perceptron , LogisticRegression , LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.decomposition import PCA
# for i in range(len(x.columns)) :
#     pca = PCA(n_components=i+1)
#     x_train_2 = pca.fit_transform(x_train)
#     x_test_2 = pca.transform(x_test)
#     model = RandomForestClassifier()
#     model.fit(x_train_2,y_train)
#     result = model.score(x_test_2,y_test)
#     print('='*50)
#     print('n_components = ', i+1 ,'result',result)
#     print('='*50)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
lda = LinearDiscriminantAnalysis(n_components=2)
x_train2 = lda.fit_transform(x_train,y_train)
x_test2 = lda.transform(x_test)
#2
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor

# model = RandomForestRegressor()
model = RandomForestClassifier()

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
print(x.shape)

evr = lda.explained_variance_ratio_

evr_cumsum = np.cumsum(evr)   
print(evr_cumsum)


# ==================================================
# n_components =  1 result 0.8611111111111112
# ==================================================
# n_components =  2 result 0.9444444444444444
# ==================================================
# n_components =  3 result 1.0
# ==================================================
# n_components =  4 result 1.0


# model.score 1.0
# (120, 4)
# [0.73398261 0.94821331 0.99546419 1.        ]


# model.score 1.0
# (120, 4)
# [0.99218869 1.        ]