# Principal component analysis = pca = 주성분 분석 // 주로 차원 축소로 많이 사용됨
# 대부분의 상황에서 성능이 떨어짐 하지만 0이 많을 때 사용하면 성능이 올라갈 수 있음
# train_test_split후 스케일링 후 PCA

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
print(sk.__version__)       # 1.3.0


#1 데이터
datasets = load_iris()
x = datasets['data']    # datasets.data 랑 똑같다
y = datasets.target
print(x.shape,y.shape)          # (150, 4) (150,)

# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# pca = PCA(n_components=1)       # n_components = 선을 몇번 그을지 1 개 그으면 4개를 1개에 다 축소 , 2면 2개 그어서 2개로 축소
#                                 # 3개 그으면 3개로 축소 이런 형식으로 변환 됨 
#                                 # 사용전에 scaler를 사용해야 됨(수치가 다 달라서 scaler로 수치들의 평균을 맞춰줘야 됨)
#                                 # scaler 중 제일 많이 이용하는건 standard(일반화)를 많이 사용한다 

# x = pca.fit_transform(x)
# print(x)
# print(x.shape)

x_train, x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state=777 , stratify=y, shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


pca = PCA(n_components=3) 
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print(x_train)
print(x_train.shape)


#2 모델구성
model = RandomForestClassifier()

#3 훈련
model.fit(x_train,y_train)

#4 평가,예측
result = model.score(x_test,y_test)
print('model.score' , result)
print(x.shape)

# PCA 사용 전
# model.score 0.9333333333333333
# (150, 4)

# PCA 3
# model.score 0.9333333333333333
# (150, 3)

# PCA 2
# model.score 0.9666666666666667
# (150, 2)

# PCA 1
# model.score 0.9666666666666667
# (150, 1)