import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import VisionTextDualEncoderModel
from keras.models import Sequential
from keras.layers import Dense , Conv3D , Flatten , Dropout
from transformers import ViTModel
import torch.nn as nn

# 영상 파일이 있는 디렉토리
video_directory = 'D:/minipro//'

# 변경할 확장자
new_extension = '.avi'

# 디렉토리 내 파일 목록 가져오기
file_list = os.listdir(video_directory)

# 영상 데이터 읽기
video_data = []
for file_name in file_list:
    # 파일 경로 생성
    file_path = os.path.join(video_directory, file_name)
    
    # 파일 확장자 확인 (.avi, .mp4 등)
    if file_path.endswith('.avi') or file_path.endswith('.mp4'):
        # VideoCapture 객체 생성
        video = cv2.VideoCapture(file_path)
        
        # 프레임 단위로 영상 읽기
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
        
        # 영상 데이터 추가
        video_data.append(frames)
        
        # VideoCapture 객체 해제
        video.release()

# 영상 데이터 확인
for i, video_frames in enumerate(video_data):
    print(f"Video {i+1} frames: {len(video_frames)}")

# text 값 가져오기
text = pd.read_csv('KETI-2017-SL-Annotation-v2_1.csv')

# y 라벨 값 때기
text = text['한국어']










