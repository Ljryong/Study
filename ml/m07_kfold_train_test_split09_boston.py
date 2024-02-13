from sklearn.datasets import load_boston
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import pandas as pd
# warning 뜨는것을 없애는 방법, 하지만 아직 왜 뜨는지 모르니 보는것을 추천
import warnings
warnings.filterwarnings('ignore') 
from sklearn.svm import LinearSVR

datasets = load_boston()



x = datasets.data
y = datasets.target



x_train , x_test, y_train, y_test = train_test_split(x,y,test_size= 0.3 , stratify= y  ,random_state= 1234 , shuffle=True)
#2 모델구성
from sklearn.model_selection import StratifiedKFold , GridSearchCV , RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import time
kfold = StratifiedKFold(n_splits= 3 , shuffle=True , random_state= 1234 )


#2 모델
model = RandomForestRegressor()
                     

from sklearn.model_selection import StratifiedKFold , cross_val_predict , cross_val_score
kfold = StratifiedKFold(n_splits=5 , shuffle=True , random_state=0)

score = cross_val_score(model , x_train, y_train, cv=kfold  )

print('Acc :',score ,'\n 평균 acc :' , round(score[1],4) )

pred = cross_val_predict(model,x_test,y_test,cv=kfold  )

acc = r2_score(y_test,pred)

print(acc)
                  




