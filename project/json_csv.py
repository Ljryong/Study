import json
import os
import csv

def extract_name_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        name = data.get('data', [])[0].get('attributes', [])[0].get('name', '')
        return name

def extract_and_save_to_csv(folder_path, output_csv_path, start_index, end_index):
    data_list = []
    for i in range(start_index, end_index + 1):
        file_name = f"NIA_SL_WORD{i}_REAL17_F_morpheme.json"
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            name = extract_name_from_json(file_path)
            data_list.append([f"WORD{i}", name])

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Index', 'Name'])
        writer.writerows(data_list)

# 함수 호출: 폴더 내의 JSON 파일에서 "name" 키 다음에 오는 문자열을 CSV 파일에 저장
extract_and_save_to_csv('D:\\수어 영상\\2.Validation\\morpheme\\17\\', 'output.csv', 1501, 1576)
