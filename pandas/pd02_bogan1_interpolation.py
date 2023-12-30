# interpolation = 보간법 (결측치 처리에서 많이 사용 됨)

'''
결측치 처리
1. 행 또는 열 삭제
2. 임의의 값
평균 : mean
중위(이상치가 있을 경우 중위값으로 하는게 좋다) : median
0 : fillna
앞의 값 : ffill(front)
뒤의 값 : bfill(back)
특정값 : 777 (특정값을 넣을때엔 조건을 같이 넣는게 좋다)
등등

3. 보간 : interpolate
4. 모델 : predict(결측치를 y로 두고 훈련시켜서 결측치를 찾고 다시 다른 모델로 y를 찾는다)
5. 부스팅 계열 : 통상 결측치 이상치에 대해 자유롭다(결측치 처리를 하는게 성능이 좋기는 함)(scaler 안해도 됨 하지만 안한게 좋은건 아님)

'''



import pandas as pd
from datetime import datetime
import numpy as np

dates = ['2/16/2024','2/17/2024','2/18/2024',
         '2/19/2024','2/20/2024','2/21/2024',]

dates = pd.to_datetime(dates)
print(dates)

# DatetimeIndex(['2024-02-16', '2024-02-17', '2024-02-18', '2024-02-19',
#                '2024-02-20', '2024-02-21'],
#               dtype='datetime64[ns]', freq=None)

print('=================================')
ts = pd.Series([2,np.nan,np.nan,
                8,10,np.nan] , index = dates )
print(ts)
# 2024-02-16     2.0
# 2024-02-17     NaN
# 2024-02-18     NaN
# 2024-02-19     8.0
# 2024-02-20    10.0
# 2024-02-21     NaN
print('=================================')
ts = ts.interpolate()
print(ts)
# 2024-02-16     2.0
# 2024-02-17     4.0
# 2024-02-18     6.0
# 2024-02-19     8.0
# 2024-02-20    10.0
# 2024-02-21    10.0























