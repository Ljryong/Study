# FI = feature importances

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from xgboost import XGBClassifier
import pandas as pd

class CustomXGBClassifier(XGBClassifier) :              # 상속 XGBClassifier
    def __str__(self):
        return 'XGBClassifier()'

# 해석 CustomXGBClassifier()을 쓰면 XGBClassifier() 라고 나온다

# aaa = CustomXGBClassifier()
# print(aaa)              # XGBClassifier()
# aaa 는 보통 인스턴스라고 부른다 // 이건 보여주기 위해서 쓴것

#1
# x,y = load_iris(return_X_y=True)
datasets = load_iris()
x = datasets.data
y = datasets.target

df = pd.DataFrame(x , columns = datasets.feature_names)
print(df)
df['target(Y)'] = y
print(df)

print('=================== 상관계수 히트맵 =====================')
print(df.corr())                                 # correlation
#                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target(Y)
# sepal length (cm)           1.000000         -0.117570           0.871754          0.817941   0.782561
# sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126  -0.426658
# petal length (cm)           0.871754         -0.428440           1.000000          0.962865   0.949035
# petal width (cm)            0.817941         -0.366126           0.962865          1.000000   0.956547
# target(Y)                   0.782561         -0.426658           0.949035          0.956547   1.000000

# -0.1 이 0보다 좋다
# 다중공섬선?
# 상관관계가 너무 높은 애들을 가지쳐주는게 성능이 좋을수도 있다. 높은애들만 있으면 걔들한테만 과적합되어서 상관관계가 없는 애들이 피해를 봄
# y값과 상관관계가 있는게 좋은 데이터 , x값과 상관관계가 있는게 애매한 데이터


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
print(sns.__version__)
print(matplotlib.__version__)               # 3.8.0
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), 
            square=True,    
            annot=True,                       # 표안에 수치 명시
            cbar=True)                        # 사이드 바
plt.show()


# matplotlib.__version__ 이 3.8.0에서 수치가 잘 안나옴
# 3.7.2 수치가 잘나와서 롤백





