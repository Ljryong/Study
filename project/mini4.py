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
from transformers import ViTModel, ViTTokenizer

# 영상 파일이 있는 디렉토리
video_directory = 'D:/minipro//'

# 변경할 확장자
new_extension = '.avi'

# 디렉토리 내 파일 목록 가져오기
file_list = os.listdir(video_directory)

# text 값 가져오기
text = pd.read_csv('KETI-2017-SL-Annotation-v2_1.csv')

# y 라벨 값 때기
text = text['한국어']

for file_name in file_list:
    # 파일의 확장자 확인
    if file_name.endswith('.mov'):
        # 기존 확장자를 새 확장자로 변경하여 파일 이름 수정
        new_file_name = os.path.splitext(file_name)[0] + new_extension
        # 파일 이름 변경
        os.rename(os.path.join(video_directory, file_name), os.path.join(video_directory, new_file_name))
        
for file_name in file_list:
    # 파일의 확장자 확인
    if file_name.endswith('.mp4'):
        # 기존 확장자를 새 확장자로 변경하여 파일 이름 수정
        new_file_name = os.path.splitext(file_name)[0] + new_extension
        # 파일 이름 변경
        os.rename(os.path.join(video_directory, file_name), os.path.join(video_directory, new_file_name))

for file_name in file_list:
    # 파일의 확장자 확인
    if file_name.endswith('.mts'):
        # 기존 확장자를 새 확장자로 변경하여 파일 이름 수정
        new_file_name = os.path.splitext(file_name)[0] + new_extension
        # 파일 이름 변경
        os.rename(os.path.join(video_directory, file_name), os.path.join(video_directory, new_file_name))

print('모든 확장자 avi로 변경')

# 데이터 전처리
tokenizer = ViTTokenizer.from_pretrained('google/vit-base-patch16-224-in21k')
image_path = "image.jpg"
image = Image.open(image_path)
inputs = tokenizer(image, return_tensors="pt")

# 모델 생성
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# 모델 훈련
# 이 부분은 새로운 데이터셋을 사용하는 경우에 해당합니다.

# 모델 평가 및 활용
outputs = model(**inputs)
text_output = outputs.last_hidden_state