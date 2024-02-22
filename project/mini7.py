import pandas as pd

# CSV 파일을 읽어옵니다.
path = 'c:/Study/'
data = pd.read_csv(path + 'KETI-2017-SL-Annotation-v2_1.csv')

old_part = '.MOV'
new_part = '.MP4'
data['파일명'] = data['파일명'].str.replace(old_part, new_part)

old_part = '.avi'
data['파일명'] = data['파일명'].str.replace(old_part, new_part)

old_part = '.MTS'
data['파일명'] = data['파일명'].str.replace(old_part, new_part)


# 변경된 데이터를 새로운 CSV 파일로 저장합니다.
data.to_csv(path + 'new_csv_file.csv', index=False)