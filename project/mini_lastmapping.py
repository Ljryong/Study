from keras.models import Sequential , Model
from keras.layers import Dense , Conv3D , Flatten , Input , GlobalMaxPooling3D, GlobalAveragePooling3D , Attention 
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import cv2
from keras.callbacks import EarlyStopping
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score
import time

#1 데이터
video_path = 'D:/minipro//'
csv_path = 'C:/Study//'

# 디렉토리 내 파일 목록 가져오기
file_list = os.listdir(video_path)
test_csv = pd.read_csv(csv_path + 'new_csv_file.csv')

# 영상 파일 이름을 기준으로 CSV 파일과 매핑
video_data = []
for file_name in file_list:
    # CSV 파일에서 해당 파일 이름에 해당하는 데이터 가져오기
    video_info = test_csv[test_csv['파일명'] == file_name]
    if not video_info.empty:
        # 필요한 정보 가져오기 예시: 레이블 가져오기
        label = video_info['한국어'].values[0]
        # 영상 파일의 경로와 함께 데이터 저장
        video_data.append((os.path.join(video_path, file_name), label))

# 영상 데이터와 레이블 매핑 결과 출력
for video_path, label in video_data:
    print(f"영상 파일: {video_path}, 레이블: {label}")

print(video_data)
print('매핑 끝')


test_csv = test_csv['한국어']

print(test_csv)