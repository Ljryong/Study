import torch
from torch.utils.data import DataLoader #,  Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
from torchvision.datasets import  VisionDataset
import cv2

# 영상 파일이 있는 디렉토리
video_directory = 'D:/minipro//'

# 변경할 확장자
new_extension = '.avi'

# 디렉토리 내 파일 목록 가져오기
file_list = os.listdir(video_directory)

# 데이터셋 클래스 정의
class CustomDataset(VisionDataset):
    def __init__(self, image_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.data = csv_file
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx,0]  # CSV 파일에서 이미지 파일 이름 가져오기
        image_path = os.path.join(self.image_dir, image_name)
        # 비디오 파일 열기
        cap = cv2.VideoCapture('D:/minipro/KETI_SL_0000004169.MP4')

        # 비디오 파일에서 프레임 읽기
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 이미지로 변환
                image = Image.fromarray(frame_rgb)
                # 이미지 전처리
                if self.transform:
                    image = self.transform(image)
                # 레이블 가져오기
                label = self.data.iloc[idx, 1]
                # 이미지와 레이블 반환
                return image, label
            else: 
                break

        # 비디오 파일 닫기
        cap.release()

        if self.transform:
            image = self.transform(image)

        label = self.data.iloc[idx,1]  # CSV 파일에서 레이블 가져오기

        return image, label

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # 이미지 크기 조정
    transforms.ToTensor(),           # 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 이미지 정규화
])

# 데이터셋 및 데이터로더 생성
image_dir = 'D:/minipro/'
csv_file = pd.read_csv('C:/Study/new_csv_file.csv')


new_csv_file = pd.DataFrame(columns=['파일명','한국어'] , dtype = str )
new_csv_file['한국어'] = csv_file['한국어']
new_csv_file['파일명'] = csv_file['파일명']
# print(new_csv_file.columns)
print(new_csv_file.head)
custom_dataset = CustomDataset(image_dir, new_csv_file, transform=transform)
batch_size = 64
custom_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# DataLoader에서 데이터 매핑 수행
for images, labels in custom_loader:
    # images와 labels는 모델의 입력과 정답에 해당합니다.
    # 이 데이터를 모델에 전달하여 학습 또는 추론을 수행합니다.
    pass