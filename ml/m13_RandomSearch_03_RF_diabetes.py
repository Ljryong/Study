from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split , KFold , cross_val_score , cross_val_predict
from sklearn.metrics import r2_score , accuracy_score
from keras.models import Sequential , load_model
from keras.layers import Dense
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVR

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)          # (442, 10) (442,)
print(datasets.feature_names)   #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x_train , x_test , y_train ,y_test = train_test_split(x,y,random_state=123 , test_size=0.3 , shuffle=True )

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#2 모델구성
from sklearn.model_selection import StratifiedKFold , GridSearchCV , RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
kfold = StratifiedKFold(n_splits= 3 , shuffle=True , random_state= 1234 )

parameters =[
    {'n_estimators' : [100,200] ,'max_depth':[6,10,12],'min_samples_leaf' : [3,10]},
    {'max_depth': [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3,5,7,10],'min_samples_split' : [2,3,5,10]},
    {'min_samples_split' : [2,3,5,10] },
    {'n_jobs' : [-1,2,4], 'min_samples_split' : [2,3,5,10]}
]
#2 모델

model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold ,
                    verbose=1,
                    refit=True,
                    n_jobs= -1 )

# model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold ,
#                                 verbose=1,
#                                 refit=True,
#                                 n_jobs= -1 
                              #   random_state=66, 
                              #   n_iter= 10 )


#3 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4 평가
y_predict = model.predict(x_test)
print('accuracy_score' , accuracy_score(y_test,y_predict))

print(accuracy_score(y_test,y_predict))

print('시간 : ' , round(end - start,2))
# print(hist)
# plt.title('diabetes loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()

# plt.show()




# 0.4282132267148565 = r2 
# 3493.03271484375 = loss

# GridSearchCV
# Fitting 3 folds for each of 60 candidates, totalling 180 fits
# accuracy_score 0.015037593984962405
# 0.015037593984962405
# 시간 :  3.15

# RandomizedSearchCV
# Fitting 3 folds for each of 10 candidates, totalling 30 fits
# accuracy_score 0.007518796992481203
# 0.007518796992481203
# 시간 :  1.85

