# Principal component analysis = pca = 주성분 분석 // 주로 차원 축소로 많이 사용됨
# 대부분의 상황에서 성능이 떨어짐 하지만 0이 많을 때 사용하면 성능이 올라갈 수 있음
# train_test_split 전 스케일링 후 PCA


from sklearn.datasets import load_diabetes , load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier ,RandomForestRegressor
print(sk.__version__)       # 1.3.0
import numpy as np


#1 데이터
datasets = load_diabetes()
datasets = load_breast_cancer()
x = datasets['data']    # datasets.data 랑 똑같다
y = datasets.target
print(x.shape,y.shape)     

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=18)       # n_components = 선을 몇번 그을지 1 개 그으면 4개를 1개에 다 축소 , 2면 2개 그어서 2개로 축소
                                # 3개 그으면 3개로 축소 이런 형식으로 변환 됨 
                                # 사용전에 scaler를 사용해야 됨(수치가 다 달라서 scaler로 수치들의 평균을 맞춰줘야 됨)
                                # scaler 중 제일 많이 이용하는건 standard(일반화)를 많이 사용한다 

x = pca.fit_transform(x)


x_train, x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state=777, shuffle=True)

#2 모델구성
model = RandomForestRegressor()
model = RandomForestClassifier()

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
print(x.shape)

evr = pca.explained_variance_ratio_
                # 설명할수있는 변화율
print(evr)
print(sum(evr))

evr_cumsum = np.cumsum(evr)         # 누적시키면서 계속 더해줌
print(evr_cumsum)

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()


# n_components = 3
# print(evr) 
# [0.40242108 0.14923197 0.12059663] pca.explained_variance_ratio_ 변화율이 얼마나 되는지
# print(sum(evr)) = 0.6722496686876478
# 변화율이 높은지 낮은지에 따라 좋은건 판단하지 못함.
# model.score 0.4494039637376369
# (442, 8)
# [0.44272026 0.63243208 0.72636371 0.79238506 0.84734274 0.88758796
#  0.9100953  0.92598254 0.93987903 0.95156881 0.961366   0.97007138
#  0.97811663 0.98335029 0.98648812 0.98915022 0.99113018 0.99288414]
# 0.99288414 = 데이터가 0.00711586 만큼 원본과 다르게 변화된것이다.
# matplotlib(시각화)을 사용하여 변화율이 낮은걸 보고 이정도면 괜찮지 않을까라는걸 확인해볼 수 있다.

