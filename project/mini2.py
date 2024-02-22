import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import VisionTextDualEncoderModel

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


class VideoDataset(Dataset):
    def __init__(self, video_directory, text_data, transform=None):
        self.video_directory = video_directory
        self.video_files = [f for f in os.listdir(video_directory) if f.endswith('.avi')]
        self.text_data = text_data
        self.transform = transform
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_directory, self.video_files[idx])
        # 영상 로드
        video = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            # 프레임 변환 (예: BGR -> RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        video.release()
        # 변환된 프레임 리스트를 텐서로 변환
        video_tensor = torch.stack([transforms.ToTensor()(frame) for frame in frames])
        if self.transform:
            video_tensor = self.transform(video_tensor)
        # 텍스트 데이터 가져오기
        text = self.text_data[idx].strip()  # 양쪽 공백 제거
        return video_tensor, text

# 데이터셋 및 데이터로더 생성
transform = transforms.Compose([transforms.Resize((1920, 1024 )),  # 예시로 이미지 크기를 224x224로 조정
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])  # 이미지 정규화
train_dataset = VideoDataset(video_directory, y_train, transform=transform)
test_dataset = VideoDataset(video_directory, y_test, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 트랜스포머 모델 정의 및 초기화
# 여기에 트랜스포머 모델을 정의하고 초기화하는 코드를 추가하세요
# 예: model = YourTransformerModel()



# 모델 훈련
for epoch in range(num_epochs):
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()






