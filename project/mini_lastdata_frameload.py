import time
import os
import cv2
import pandas as pd
import numpy as np

#1 데이터
video_path = 'D:/minipro//'
csv_path = 'C:/Study//'

# 디렉토리 내 파일 목록 가져오기
file_list = os.listdir(video_path)
test_csv = pd.read_csv(csv_path + 'new_csv_file.csv')

start = time.time()
# 프레임을 저장할 리스트 생성
frames = []

# 각 영상의 프레임 수를 저장할 리스트 생성
video_lengths = []

# 비디오 파일을 프레임 단위로 읽어들임
for file_name in file_list:
    file_path = os.path.join(video_path, file_name)
    cap = cv2.VideoCapture(file_path)
    
    # 프레임 수 초기화
    num_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            num_frames += 1
        else:
            break
    
    # 현재 비디오의 프레임 수를 저장
    video_lengths.append(num_frames)
    
    cap.release()
    
end = time.time()

print('비디오 읽기 끝')
print('비디오 읽기 시간' , end - start)

# 패딩 적용하지 않고 x 값을 생성
x_values = np.array(frames, dtype='float32')

print(x_values.shape)

