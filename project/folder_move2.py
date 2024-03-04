import os
import shutil

source_directory = "D:\수어 영상\\1.Training\[원천]23_real_word_video\\12"
target_directory = "C:\\3TB\\video\\"

# source_directory에 있는 모든 MP4 파일에 대해 반복
for file_name in os.listdir(source_directory):
    if file_name.endswith(".mp4"):
        # 파일 이름에서 "WORD" 다음의 숫자를 추출
        word_index = file_name.find("WORD") + len("WORD")
        number_str = ""
        for char in file_name[word_index:]:
            if char.isdigit():
                number_str += char
            else:
                break

        # 추출된 숫자를 폴더 이름으로 사용하여 해당 폴더가 있는지 확인
        folder_path = os.path.join(target_directory, number_str)
        if os.path.exists(folder_path):
            # 파일 이름에 "F"가 포함되어 있는지 확인하여 이동할지 결정
            if "F" in file_name:
                # 대상 파일이 이미 대상 폴더에 존재하는지 확인
                target_file_path = os.path.join(folder_path, file_name)
                if os.path.exists(target_file_path):
                    print(f"'{file_name}' 파일은 이미 '{folder_path}' 폴더에 존재합니다. 이동되지 않았습니다.")
                else:
                    # 파일을 해당 폴더로 이동
                    shutil.move(os.path.join(source_directory, file_name), folder_path)
                    print(f"'{file_name}' 파일을 '{folder_path}' 폴더로 이동했습니다.")
            else:
                print(f"'{file_name}' 파일은 'F'를 포함하지 않아 이동되지 않았습니다.")
        else:
            print(f"'{folder_path}' 폴더가 존재하지 않아 '{file_name}' 파일을 이동할 수 없습니다.")
