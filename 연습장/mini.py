import pandas as pd

# 엑셀 파일 경로
excel_file = 'KETI-2018-SL-Annotation-v1.xlsx'

# 엑셀 파일 읽기
df = pd.read_excel('C:/_data//'+excel_file)

# CSV 파일로 변환하여 저장
csv_file = 'KETI-2018-SL-Annotation-v1.csv'
df.to_csv(csv_file, index=False)