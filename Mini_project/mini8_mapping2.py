import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
import os
import pandas as pd


class VideoDataset(Dataset):
    def __init__(self, video_path, csv_file, transform=None):
        self.video_path = video_path
        self.csv_file = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.video_path)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_path[idx])
        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV에서 읽은 이미지를 RGB 형식으로 변환
                if self.transform:
                    frame = self.transform(frame)  # 이미지 전처리
                frames.append(frame)
            else:
                break
        cap.release()

        if not frames:  # frames가 비어 있는 경우, 즉 비디오를 읽지 못한 경우
            raise ValueError(f"Unable to read frames from video: {self.video_path[idx]}")
        
        # 리스트에 있는 모든 프레임을 하나의 텐서로 결합
        video_tensor = torch.stack(frames)
        
        label = self.csv_file[idx]

        return video_tensor, label

# 데이터 전처리를 위한 변환 정의
transform = transforms.Compose([
    transforms.ToPILImage(),  # OpenCV 이미지를 PIL 이미지로 변환
    transforms.Resize((100, 100)),  # 이미지 크기 조정
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 이미지 정규화
])

# 디렉토리 내 파일 목록 가져오기
video_directory = 'D:/minipro//'
file_list = os.listdir(video_directory)

# 영상 파일 경로와 레이블
# video_files = ['video1.mp4', 'video2.mp4', 'video3.mp4']
# csv_file = ['label1', 'label2', 'label3']
csv_file = pd.read_csv('C:/Study/new_csv_file.csv')


new_csv_file = pd.DataFrame(columns=['파일명','한국어'] , dtype = str )
new_csv_file['한국어'] = csv_file['한국어']
new_csv_file['파일명'] = csv_file['파일명']

# 데이터셋 인스턴스 생성
video_dataset = VideoDataset(file_list, csv_file, transform=transform)

# 데이터로더 생성
batch_size = 64
video_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)

# 데이터로더에서 데이터 매핑 수행
for videos, labels in video_loader:
    # videos와 labels는 모델의 입력과 정답에 해당
    # 이 데이터를 모델에 전달하여 학습 또는 추론 수행
    pass