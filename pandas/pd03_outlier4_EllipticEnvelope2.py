# 정상치 범위 찾기(이상치 찾기)

import numpy as np
aaa = np.array([[-10,2,3,4,5
                ,6,7,8,9,10,
                11,12,50],
               [100,200,-30,400,500
                ,600,-70000,800,900,1000,
                210,420,350]]).T         # .T 를 사용하여 바꿔줌 (13,2) (2,13)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1)           
# contamination=.3 이면 데이터의 30% 가 이상치로 간주
# 얼마나 많은 데이터 포인트를 이상치로 취급할지 결정
# 2개의 열을 1개로 인식함 
# 그래서 컬럼별로 진행해야 함

outliers.fit(aaa)
results = outliers.predict(aaa)    
print(results)              # -1이 이상치 
# [-1  1  1  1  1  1  1  1  1  1  1  1 -1]