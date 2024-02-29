import os
import pandas as pd

# 1. CSV 파일 읽기
df = pd.read_csv('C:/Study/new_csv_file.csv')

# 2. 파일명을 기반으로 데이터 찾기
# 예시: 파일명에서 확장자를 제거하여 파일 제목 추출
file_title = os.path.splitext(os.path.basename('C:/Study/new_csv_file.csv'))[0]

# 3. 해당하는 행 찾기
# 예시: 'Title' 열의 값이 파일 제목과 일치하는 행 찾기
matching_row = df[df['한국어'] == file_title]

# 4. 내용 변경
# 예시: 찾은 행의 내용을 변경
# 예시로 모든 내용을 'New Content'로 변경
matching_row['Content'] = 'New Content'

# 5. 변경된 데이터를 새로운 CSV 파일로 저장
df.to_csv('updated_file.csv', index=False)