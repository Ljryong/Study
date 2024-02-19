# 정상치 범위 찾기(이상치 찾기)

import numpy as np
aaa = np.array([-10,2,3,4,5,6,7,8,9,10,11,12,50])

print(aaa.shape)        # (13,)

def outliers(data_out):
    quartile_1 , q2 , quartile_3 = np.percentile(data_out,[25,50,75])       # 25,50,75 퍼센트로 나눔
    print('1사 분위 :' , quartile_1 )
    print('q2 :' , q2 )
    print('3사 분위 :' , quartile_3 )
    iqr = quartile_3 - quartile_1                       
    # 이상치는 보통의 값을 벗어난 것인데 이상치는 엄청 크거나 엄청 작거나 둘중 하나이다
    # 이런걸 방지하기위해서 상위25%과 하쉬25%를 버리고 나머지 50%를 가져온다.
    # 가운데 데이터들은 보통 정상적인 데이터라고 판단(아닐수도 잇음) 
    print('iqr :' , iqr)
    lower_bound = quartile_1 - (iqr * 1.5)              
    # 1.5가 아니여도 되는데 통상 1.5가 제일 좋음
    # 로우 = 4 - (6 * 1.5) = 4 - 9 = -5 여기까지의 데이터를 이상치가 아니라고 판단한다
    upper_bound = quartile_3 + (iqr * 1.5)
    # 하이 = 10 + (6 * 1.5) = 10 + 9 = 19 여기까지의 데이터를 이상치가 아니라고 판단
    return np.where((data_out>upper_bound) | (data_out<lower_bound))        # | python 함수에서 or 이랑 같은 뜻이다
    # 2가지 조건중에 한개라도 만족하는걸 빼냄 19큰거 -5보다 작은걸 빼내라
    # 뽑으면 위치값 0 , 12 의 값이 이상치라고 나옴
outliers_loc = outliers(aaa)
print('이상치의 위치 :' , outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

# 1사 분위 : 4.0        25%
# q2 : 7.0              50%
# 3사 분위 : 10.0        75%
# iqr : 4.0
# 이상치의 위치 : (array([ 0, 12], dtype=int64),)

# 이상치라고 찾은 애들을 수치를 바꿔서 넣어주던가 삭제시키던가 그건 자신이 판단


