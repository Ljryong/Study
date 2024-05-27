import os

def create_folders(start, end, base_path='./'):
    """
    start: int, 시작 번호
    end: int, 끝 번호
    base_path: str, 기본 경로 (기본값은 현재 디렉토리)

    지정된 범위 내에서 폴더를 생성합니다.
    """
    for i in range(start, end + 1501):
        folder_name = os.path.join(base_path, str(i))
        os.makedirs(folder_name, exist_ok=True)

# 사용자의 홈 디렉토리에 1부터 76까지의 폴더를 생성합니다.
create_folders(1501, 1576, base_path='C:\\3TB\\video\\')  # 'username'을 실제 사용자 이름으로 변경해주세요.
