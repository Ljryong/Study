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

x_train, x_test , y_train , y_test = train_test_split(file_list, text , test_size=0.2 , shuffle=True , random_state= 1234 )


#2 모델
# vision_model = 

# text_model = 

# model = VisionTextDualEncoderModel(
#                                     vision_model = vision_model , 
#                                     text_model =  text_model 
#                                    )

# 모델 초기화
model = VisionTextDualEncoderModel.from_pretrained("bert-base-uncased")

# 이미지와 텍스트 입력 데이터 생성
image_input = x_train
text_input = y_train

# 이미지와 텍스트를 모델에 입력하여 임베딩 생성
image_embedding = model.get_image_embedding(image_input)
text_embedding = model.get_text_embedding(text_input)

# 이미지와 텍스트 간의 유사성 계산
similarity_score = torch.cosine_similarity(image_embedding, text_embedding)
print("Similarity Score:", similarity_score.item())




model.summary()
