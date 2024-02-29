import os
import random
import shutil

# 폴더 내 파일 개수
total_files = 10480

# 비율 설정 (예: 60%, 20%, 20%)
ratios = [0.7, 0.2, 0.1]

# 파일 목록 가져오기
video_folder = 'D:/minipro//'
file_list = os.listdir(video_folder)

# 파일 목록 섞기
random.shuffle(file_list)

# 파일 그룹 나누기
file_groups = []
start_idx = 0
for ratio in ratios:
    end_idx = start_idx + int(total_files * ratio)
    file_groups.append(file_list[start_idx:end_idx])
    start_idx = end_idx

# 파일 그룹을 폴더로 이동
for i, files in enumerate(file_groups, start=1):
    folder_path = os.path.join(video_folder, f'폴더{i}')
    os.makedirs(folder_path, exist_ok=True)
    for file in files:
        src_path = os.path.join(video_folder, file)
        dst_path = os.path.join(folder_path, file)
        shutil.move(src_path, dst_path)

print("파일을 비율에 맞게 분할하였습니다.")
