from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC


#1

datasets = load_wine()
x= datasets.data
y= datasets.target

print(x.shape,y.shape)      # (178, 13) (178,)
print(pd.value_counts(y))   # 1    71 , 0    59 , 2    48
print(datasets.feature_names)
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
#  'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 
#  'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

# x = np.delete(x,(1,3,10),axis=1)

x = pd.DataFrame(x , columns = datasets.feature_names )
# x = x.drop(['malic_acid','alcalinity_of_ash', 'hue'],axis=1)

print(np.unique(y))

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.3, random_state= 0 ,shuffle=True, stratify = y)

es = EarlyStopping(monitor='val_loss', mode = 'min' , verbose= 1 ,patience=20 ,restore_best_weights=True)

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
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3 훈련
model.fit(x_train2,y_train)

#4 평가,예측
result = model.score(x_test2,y_test)
print('model.score' , result)
print(x.shape)

evr = lda.explained_variance_ratio_

evr_cumsum = np.cumsum(evr)   
print(evr_cumsum)

# (178, 13)
# [0.36951469 0.55386396 0.67201555 0.74535807 0.80957914 0.86009639]

# ====================================================================================================
# n_components =  1 result 0.6666666666666666
# ====================================================================================================
# n_components =  2 result 0.7037037037037037
# ====================================================================================================
# n_components =  3 result 0.7222222222222222
# ====================================================================================================
# n_components =  4 result 0.9814814814814815
# ====================================================================================================
# n_components =  5 result 0.9629629629629629
# ====================================================================================================
# n_components =  6 result 0.9814814814814815
# ====================================================================================================
# n_components =  7 result 1.0
# ====================================================================================================
# n_components =  8 result 1.0
# ====================================================================================================
# n_components =  9 result 1.0
# ====================================================================================================
# n_components =  10 result 1.0
# ====================================================================================================
# n_components =  11 result 1.0
# ====================================================================================================
# n_components =  12 result 1.0
# ====================================================================================================
# n_components =  13 result 0.9814814814814815


# model.score 0.9629629629629629
# (178, 13)
# [0.66162655 1.        ]
