import pandas as pd

# 엑셀 파일을 읽어옵니다. Excel 파일의 경로와 파일 이름을 제공해야 합니다.
path = 'c:/_data//'
excel_file_path = '20240219_062134_excel.xlsx'  # 예시 파일 경로
df = pd.read_excel(path + excel_file_path)

# CSV 파일로 저장합니다. CSV 파일의 경로와 파일 이름을 제공해야 합니다.
csv_file_path = 'example.csv'  # 저장할 CSV 파일 경로 및 이름
df.to_csv(csv_file_path, index=False)