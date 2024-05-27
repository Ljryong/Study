import os

def delete_real18_mp4_files(folder_path):
    """
    folder_path: str, 검색 대상이 되는 최상위 폴더 경로

    folder_path와 그 하위 폴더에서 이름에 'REAL18'이 포함된 .mp4 파일을 모두 삭제합니다.
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mp4') and 'REAL12' in file:
                try:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"{file_path} 삭제 완료")
                except OSError as e:
                    print(f"{file_path} 삭제 실패: {e}")

# 특정 폴더 및 그 하위 폴더에서 이름에 'REAL18'이 포함된 .mp4 파일을 삭제합니다.
delete_real18_mp4_files('C:\\3TB\\video')  # 경로를 적절히 수정해주세요.