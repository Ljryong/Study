import pandas as pd

data = [
    ['삼성','1000','2000'],
    ['현대','1100','3000'],
    ['LG','2000','500'],
    ['아모레','3500','6000'],
    ['네이버','100','1500'],          
]

index = ['031','059','033','045','023']             # index = 연산할때 쓰는 데이터가 X
columns = ['종목명','시가','종가']                   # columns = 연산할때 쓰는 데이터가 X

df = pd.DataFrame(data=data , index=index , columns=columns)
# print(df)
print('=============================='*5)

aaa = df['시가']>='1100'
print(aaa)
print(df[aaa])
print(df.loc[aaa])
# print(df.iloc[aaa])       error iloc 에는 위치값이 와야 함
print(df[df['시가']>='1100'])
#  pandas 에서 어떤 조건식을 뽑을라면 df 안에 df[조건식]을 넣어주면 됨

df_2 = df.loc[df['시가']>='1100']        # df.loc[df.loc['시가']>='1100'] 의 행을 뽑아냄
print(df_2)
# 059   현대  1100  3000
# 033   LG  2000   500
# 045  아모레  3500  6000
# df_2 = df.loc[df.loc['031','시가']>='1100']        # df.loc[df.loc['시가']>='1100'] 의 행을 뽑아냄

df_3 = df.loc[df['시가']>='1100','종가']     
# df.loc[df.loc['시가']>='1100','종가']에서 df.loc['시가']>='1100'의 행 , '종가' 열을 뽑아냄
print(df_3)
# 059    3000
# 033     500
# 045    6000

df_3 = df[df['시가']>='1100']['종가']     
print(df_3)
print(df.loc[df['시가']>='1100']['종가'])
print(df.loc[df['시가']>='1100','종가'])

# python이 앞에서 부터 숫자를 읽어가지고 90 < 100 을 비교하면 틀리다고 나온다.
# 이런 상황을 방지하기 위해서 문자를 int를 쓰든 encoder를 써서 숫자형태로 바꾼다음 비교하는게 좋다.


