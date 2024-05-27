# import pandas as pd

# # CSV 파일 불러오기
# df = pd.read_csv("c:/study/new3_csv_file.csv")

# # '타입(단어/문장)' 열이 문장인 행 삭제
# df = df[df['타입(단어/문장)'] == '단어']

# # '한국어' 열의 값이 중복된 행 제거
# df.drop_duplicates(subset=['한국어'], keep='first', inplace=True)

# # 수정된 DataFrame을 CSV 파일로 저장
# df.to_csv("new4_csv_file.csv", index=False)

import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv("c:/study/new_csv_file.csv")

# '방향' 열이 '측면'인 행 삭제
df = df[df['방향'] != '측면']

# '타입(단어/문장)' 열이 문장인 행 삭제
df = df[df['타입(단어/문장)'] == '단어']

# '한국어' 열의 값이 중복된 행 제거
df.drop_duplicates(subset=['한국어'], keep='first', inplace=True)

# 수정된 DataFrame을 CSV 파일로 저장
df.to_csv("new2_csv_file.csv", index=False)

