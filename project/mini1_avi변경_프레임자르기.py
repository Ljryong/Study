import cv2
import os

# 영상 파일이 있는 디렉토리
video_directory = 'D:/mini//'

# 변경할 확장자
new_extension = '.avi'

# 디렉토리 내 파일 목록 가져오기
file_list = os.listdir(video_directory)

import os

def change_extension(directory, old_ext, new_ext):
    # 디렉토리 내 파일 목록 가져오기
    file_list = os.listdir(directory)
    
    # 각 파일에 대해 반복
    for file_name in file_list:
        # 파일의 확장자 확인
        if file_name.endswith(old_ext):
            # 기존 확장자를 새 확장자로 변경하여 파일 이름 수정
            new_file_name = os.path.splitext(file_name)[0] + new_ext
            # 파일 이름 변경
            os.rename(os.path.join(directory, file_name), os.path.join(directory, new_file_name))


video_directory = 'D:/mini/'
old_extension = ['.MOV','.avi','.MTS']
new_extension = '.MP4'
for old_extension in old_extension :
    change_extension(video_directory, old_extension, new_extension)



print('모든 확장자 avi로 변경')

# 이미지를 저장할 디렉토리 설정
output_directory = 'D:/_minipro_image//'
os.makedirs(output_directory, exist_ok=True)

# 영상 파일 리스트 가져오기
video_files = [f for f in os.listdir(video_directory) if f.endswith('.avi')]

for video_file in video_files:
    # VideoCapture 객체 생성
    video_path = os.path.join(video_directory, video_file)
    video = cv2.VideoCapture(video_path)

    # 프레임 단위로 영상을 읽어서 이미지로 저장
    frame_count = 0
    while True:
        ret, frame = video.read()

        if not ret:
            break

        # 이미지 저장
        image_path = os.path.join(output_directory, f'{video_file}_frame_{frame_count}.jpg')
        cv2.imwrite(image_path, frame)

        frame_count += 1

    # VideoCapture 객체 해제
    video.release()

print('영상 프레임으로 자르기 완료')














