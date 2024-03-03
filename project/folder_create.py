import os
import shutil

# 대상 디렉토리 경로 지정
target_directory = "C:\\3TB\\video\\17\\"

# 17 폴더 안의 모든 파일 목록 가져오기
files = os.listdir(target_directory)

# 모든 파일에 대해 반복
for file_name in files:
    # mp4 파일만 처리
    if file_name.endswith(".mp4"):
        # "WORD" 다음의 숫자를 추출하여 폴더 이름으로 사용
        word_index = file_name.find("WORD") + len("WORD")
        number_str = ""
        for char in file_name[word_index:]:
            if char.isdigit():
                number_str += char
            else:
                break

        # 추출된 숫자를 폴더 이름으로 사용하여 폴더 경로 설정
        folder_name = number_str
        folder_path = os.path.join(target_directory, folder_name)

        # 해당 폴더가 없으면 생성
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 파일을 해당 폴더로 이동
        new_file_name = file_name.replace(f"WORD{number_str}", "")  # "WORD"와 숫자 부분을 제거한 새로운 파일 이름
        shutil.move(os.path.join(target_directory, file_name), os.path.join(folder_path, new_file_name))

        print(f"파일 '{file_name}'을(를) '{folder_path}' 폴더에 이동했습니다.")
