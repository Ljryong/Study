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
x = x.drop(['malic_acid','alcalinity_of_ash', 'hue'],axis=1)


x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.3, random_state= 0 ,shuffle=True, stratify = y)

es = EarlyStopping(monitor='val_loss', mode = 'min' , verbose= 1 ,patience=20 ,restore_best_weights=True)

#2
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

models = [DecisionTreeClassifier(random_state = 777), RandomForestClassifier(random_state = 777) , 
          GradientBoostingClassifier(random_state = 777),XGBClassifier()]

############## 훈련 반복 for 문 ###################a
for model in models :
    model.fit(x_train,y_train)
    result = model.score(x_test,y_test)
    print(type(model).__name__,':',model.feature_importances_ ,result)
   # y_predict = model.predict(x_test)
    print(type(model).__name__,'result',result)
    
    
# DecisionTreeClassifier : [0.04137696 0.04580777 0.         0.         0.         0.
#  0.39714318 0.         0.         0.10565781 0.         0.
#  0.41001428] 0.9444444444444444
# DecisionTreeClassifier result 0.9444444444444444
# RandomForestClassifier : [0.09791557 0.02551823 0.01333271 0.02824455 0.02953136 0.06147196
#  0.17210385 0.01608801 0.02014884 0.15347595 0.0571904  0.11238148
#  0.21259709] 1.0
# RandomForestClassifier result 1.0
# GradientBoostingClassifier : [0.00586507 0.0220035  0.02908603 0.0087705  0.00205965 0.00171811
#  0.29335587 0.00150269 0.00940809 0.32166733 0.00587327 0.00308732
#  0.29560257] 0.9814814814814815
# GradientBoostingClassifier result 0.9814814814814815
# XGBClassifier : [0.02122318 0.02177828 0.062746   0.00966202 0.02777876 0.03206314
#  0.26798463 0.02240614 0.05812377 0.24999015 0.01020946 0.03436402
#  0.18167037] 0.9814814814814815
# XGBClassifier result 0.9814814814814815


# DecisionTreeClassifier : [0.01745058 0.         0.06973415 0.         0.39714318 0.
#  0.         0.10565781 0.         0.41001428] 0.9259259259259259
# DecisionTreeClassifier result 0.9259259259259259
# RandomForestClassifier : [0.10456059 0.01588712 0.03078189 0.07419615 0.15123828 0.01341435
#  0.02320427 0.19094911 0.16753683 0.22823139] 1.0
# RandomForestClassifier result 1.0
# GradientBoostingClassifier : [9.72908735e-03 4.45442816e-02 1.94454015e-02 8.66767141e-03
#  2.98548332e-01 2.46023837e-04 6.03251543e-03 3.20083772e-01
#  6.20711117e-03 2.86495803e-01] 0.9814814814814815
# GradientBoostingClassifier result 0.9814814814814815
# XGBClassifier : [0.02290061 0.16208498 0.01609911 0.03166954 0.2809482  0.00632456
#  0.06185188 0.20092611 0.03841175 0.1787833 ] 0.9444444444444444
# XGBClassifier result 0.9444444444444444