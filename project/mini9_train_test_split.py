import torch
from torch.utils.data import DataLoader #,  Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
from torchvision.datasets import  VisionDataset 
import cv2
from mini8_mapping import CustomDataset , transform
import numpy as np

# 영상 파일이 있는 디렉토리
video_directory = 'D:/minipro//'

# 변경할 확장자
new_extension = '.avi'

# 디렉토리 내 파일 목록 가져오기
file_list = os.listdir(video_directory)

# 데이터셋 및 데이터로더 생성
image_dir = 'D:/minipro/'
csv_file = pd.read_csv('C:/Study/new_csv_file.csv')


new_csv_file = pd.DataFrame(columns=['파일명','한국어'] , dtype = str )
new_csv_file['한국어'] = csv_file['한국어']
new_csv_file['파일명'] = csv_file['파일명']
# print(new_csv_file.head)

custom_dataset = CustomDataset(image_dir, new_csv_file, transform=transform)
batch_size = 64
custom_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# DataLoader에서 데이터 매핑 수행
for images, labels in custom_loader:
    # images와 labels는 모델의 입력과 정답에 해당합니다.
    # 이 데이터를 모델에 전달하여 학습 또는 추론을 수행합니다.
    pass


################# 매핑 끝 ######################
new_csv_file = new_csv_file['한국어']

videos = np.asarray(file_list)

# 동영상 파일을 읽어오는 리더 객체 생성
# video_reader = imageio.get_reader(f)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(videos, new_csv_file, test_size=0.2, random_state=42, stratify=new_csv_file)

train_dataset = CustomDataset(image_dir, x_train, transform=transform)
val_dataset = CustomDataset(image_dir, x_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(x_train)      # (8384,)
print(x_test)       # (2096,)
print(y_train)      # (8384,)
print(y_test)       # (2096,)
