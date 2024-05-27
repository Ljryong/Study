import pandas as pd

# 첫 번째 CSV 파일 읽기
df1 = pd.read_csv("new2_csv_file.csv")

# 두 번째 CSV 파일 읽기
df2 = pd.read_csv("output.csv")

# 첫 번째 CSV 파일의 '한국어' 열에서 공통된 값만 가져오기
common_values = set(df1['한국어']).intersection(df2['Name'])

# 공통된 값이 있는 행 추출
common_rows = df1[df1['한국어'].isin(common_values)]

# 결과 출력
print(common_rows)


# 형     
# 할아버지  
# 할머니   
# 자살    
# 오빠    
# 연기    
# 엄마    
# 아내    
# 딸     
# 누나    
# 남편    