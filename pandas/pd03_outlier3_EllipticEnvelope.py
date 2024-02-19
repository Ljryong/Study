# 정상치 범위 찾기(이상치 찾기)

import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])
print(aaa.shape)        # (13,)
aaa = aaa.reshape(-1,1)         # sklearn이 metrics 형태를 좋아하여 변환 (13,1)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.3)           
# contamination=.3 이면 데이터의 30% 가 이상치로 간주
# 얼마나 많은 데이터 포인트를 이상치로 취급할지 결정

outliers.fit(aaa)
results = outliers.predict(aaa)    
print(results)              # -1이 이상치 
# [-1  1  1  1  1  1  1  1  1  1  1  1 -1]