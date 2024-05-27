import json
import os

# JSON 파일이 들어있는 디렉토리 경로
directory = 'D:\수어 영상\\1.Training\[라벨]01_crowd_morpheme\morpheme\\01\\'


# 디렉토리 내의 모든 JSON 파일에 대해 반복
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        
        # JSON 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        
        # "data" 배열에서 "attributes" 배열의 "name" 키의 값을 추출
        for item in json_data["data"]:
            for attribute in item["attributes"]:
                name_value = attribute["name"]
        
                # 추출한 값 출력
                print(name_value)
                
import json
import csv
import os

# JSON 파일이 들어있는 디렉토리 경로
directory = 'D:\수어 영상\\1.Training\[라벨]01_crowd_morpheme\morpheme\\01\\'

# CSV 파일 경로
csv_file_path = 'output.csv'

# CSV 파일에 헤더를 작성할 필요가 있는지 여부
write_header = True

# CSV 파일 열기
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    # CSV writer 생성
    csv_writer = csv.writer(csv_file)
    
    # 디렉토리 내의 모든 JSON 파일에 대해 반복
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            # JSON 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
            
            # JSON 데이터에서 원하는 값을 추출하여 CSV 파일에 쓰기
            for item in json_data["data"]:
                for attribute in item["attributes"]:
                    name_value = attribute["name"]
                    
                    # CSV 파일에 데이터 작성
                    if write_header:
                        csv_writer.writerow(["Name"])  # 헤더 작성
                        write_header = False
                        
                    csv_writer.writerow([name_value])

print("CSV 파일이 생성되었습니다.")