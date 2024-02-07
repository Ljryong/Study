from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , accuracy_score
from keras.models import Sequential , load_model
from keras.layers import Dense
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape,y.shape)          # (442, 10) (442,)
print(datasets.feature_names)   #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3 , random_state= 151235 , shuffle= True )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler , RobustScaler , StandardScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2 모델구성
allAlgorisms = all_estimators(type_filter='classifier')             # 분류모델
# allAlgorisms = all_estimators(type_filter='regressor')              # 회귀모델

# print('allAlgorisms',allAlgorisms)      # 튜플형태 // 소괄호안에 담겨져있음
# print('모델의 갯수',len(allAlgorisms))          # 41 == 분류모델의 갯수 // 55 == 회귀모델의 갯수

for name, algorithm in allAlgorisms : 
    # 에러가 떳을때 밑에 print로 넘어가고 다음이 실행됨
    try:
        #2 모델
        model = algorithm()
        #3 훈련
        model.fit(x_train,y_train)
        #4 평가,예측
        acc = model.score(x_test,y_test)
        print(name, '의 정답률' , acc )
    except:
        print(name,  '은 바보 멍충이!!!')
        # continue            # 에러를 무시하고 쭉 진행됨 


# print(hist)
# plt.title('diabetes loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.grid()

# plt.show()




# 0.4282132267148565 = r2 
# 3493.03271484375 = loss


# MinMaxScaler
# Epoch 132: early stopping
# 5/5 [==============================] - 0s 0s/step - loss: 3495.9897 - mse: 3495.9897 - mae: 48.7478
# 5/5 [==============================] - 0s 0s/step
# 0.4277291969179722
# [3495.98974609375, 3495.98974609375, 48.7478141784668]

# StandardScaler
# Epoch 114: early stopping
# 5/5 [==============================] - 0s 1ms/step - loss: 3423.5649 - mse: 3423.5649 - mae: 47.5210
# 5/5 [==============================] - 0s 0s/step
# 0.43958468538922235
# [3423.56494140625, 3423.56494140625, 47.52100372314453]

# MaxAbsScaler
# Epoch 120: early stopping
# 5/5 [==============================] - 0s 0s/step - loss: 3412.5415 - mse: 3412.5415 - mae: 47.4617
# 5/5 [==============================] - 0s 0s/step
# 0.44138915418500335
# [3412.54150390625, 3412.54150390625, 47.46168899536133]

# RobustScaler
# Epoch 144: early stopping
# 5/5 [==============================] - 0s 4ms/step - loss: 3471.2327 - mse: 3471.2327 - mae: 47.7108
# 5/5 [==============================] - 0s 0s/step
# 0.4317817805060158
# [3471.232666015625, 3471.232666015625, 47.71084976196289]



# 0.4375280766636177
# 0.4375280766636177


# LinearSVR score  -0.11584717273232203
# LinearSVR predict  -0.11584717273232203
# Perceptron score  0.0
# Perceptron predict  -0.17001095003895728
# LogisticRegression score  0.0
# LogisticRegression predict  0.2471293071081715
# RandomForestClassifier score  0.007518796992481203
# RandomForestClassifier predict  -0.05620089357085534
# DecisionTreeClassifier score  0.0
# DecisionTreeClassifier predict  -0.11524009113458811
# KNeighborsClassifier score  0.007518796992481203
# KNeighborsClassifier predict  -0.9171868323838201






# AdaBoostClassifier 의 정답률 0.007518796992481203
# BaggingClassifier 의 정답률 0.007518796992481203
# BernoulliNB 의 정답률 0.0
# CalibratedClassifierCV 은 바보 멍충이!!!
# CategoricalNB 은 바보 멍충이!!!
# ClassifierChain 은 바보 멍충이!!!
# ComplementNB 은 바보 멍충이!!!
# DecisionTreeClassifier 의 정답률 0.007518796992481203
# DummyClassifier 의 정답률 0.0
# ExtraTreeClassifier 의 정답률 0.015037593984962405
# ExtraTreesClassifier 의 정답률 0.015037593984962405
# GaussianNB 의 정답률 0.007518796992481203
# GaussianProcessClassifier 의 정답률 0.015037593984962405
# GradientBoostingClassifier 의 정답률 0.007518796992481203
# HistGradientBoostingClassifier 의 정답률 0.007518796992481203
# KNeighborsClassifier 의 정답률 0.007518796992481203
# LabelPropagation 의 정답률 0.015037593984962405
# LabelSpreading 의 정답률 0.015037593984962405
# LinearDiscriminantAnalysis 의 정답률 0.0
# LinearSVC 의 정답률 0.0
# LogisticRegression 의 정답률 0.0
# LogisticRegressionCV 은 바보 멍충이!!!
# MLPClassifier 의 정답률 0.007518796992481203
# MultiOutputClassifier 은 바보 멍충이!!!
# MultinomialNB 은 바보 멍충이!!!
# NearestCentroid 의 정답률 0.0
# NuSVC 은 바보 멍충이!!!
# OneVsOneClassifier 은 바보 멍충이!!!
# OneVsRestClassifier 은 바보 멍충이!!!
# OutputCodeClassifier 은 바보 멍충이!!!
# PassiveAggressiveClassifier 의 정답률 0.007518796992481203
# Perceptron 의 정답률 0.0
# QuadraticDiscriminantAnalysis 은 바보 멍충이!!!
# RadiusNeighborsClassifier 은 바보 멍충이!!!
# RandomForestClassifier 의 정답률 0.007518796992481203
# RidgeClassifier 의 정답률 0.0
# RidgeClassifierCV 의 정답률 0.0
# SGDClassifier 의 정답률 0.0
# SVC 의 정답률 0.022556390977443608
# StackingClassifier 은 바보 멍충이!!!
# VotingClassifier 은 바보 멍충이!!!


