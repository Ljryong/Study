import numpy as np

a = np.array(range(1,11))
size = 5            # 자르는 사이즈(time steps)

def split_x(dataset,size) :         # dataset = a
    aaa=[]                          # aaa 리스트 만들기
    for i in range(len(dataset) - size+1):          # range(1~10 에서 5빼고 1 더하기 ) = 반복 횟수(행의 갯수를 알 수 있음)
                                                    # dataset의 길이에서 size를 뺀 만큼 반복합니다(6 = 행의 갯수 ). 이렇게 함으로써 생성할 부분 시계열 데이터의 개수(행)를 결정합니다
        subset = dataset[ i : ( i + size ) ]        # range(1:11)[i:(i+5)] i가 0 일때 0 : 0 + 5 = 1,2,3,4,5 를 뜻한다.
        aaa.append(subset)      # 이어 붙이기        # for문(반복문)이어서 반복된걸 append(이어 붙여준다)
    return np.array(aaa)        # 이어붙인걸 리스트 형태로 반환해서 넣어줄 틀을 만들어준다, 틀이 없으면 들어가지 않는다.

bbb = split_x(a,size)       # 위에 정의한 split_x 안에 값을 넣고 값을 함수의 형식으로 치환하기
print(bbb)                  
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
print(bbb.shape)            # (6, 5)

x = bbb[: , :-1]            # 모든 행에서 마지막 열 빼고 뽑아주세요
y = bbb[: , -1]             # 모든 행에서 마지막 열만 뽑아주세요
print(x , y)
# x = [[1 2 3 4]
#  [2 3 4 5]
#  [3 4 5 6]
#  [4 5 6 7]
#  [5 6 7 8]
#  [6 7 8 9]] 
# y = [ 5  6  7  8  9 10]
print(x.shape,y.shape)      # (6, 4) (6,)






