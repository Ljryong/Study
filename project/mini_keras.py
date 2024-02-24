from keras.models import Sequential , Model
from keras.layers import Dense , Conv3D , Flatten , Input
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

print('매핑 끝')

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

# 패딩 적용
max_length = max(video_lengths)  # 가장 긴 영상의 프레임 수를 찾음
padded_frames = pad_sequences(frames, maxlen=max_length, padding='post', dtype='float32')

y = test_csv['한국어']

# Tokenizer 객체 생성
tokenizer = Tokenizer()

# 텍스트 데이터를 토큰화하여 토큰 사전을 생성
tokenizer.fit_on_texts(y)

# 텍스트 데이터를 시퀀스로 변환
sequences = tokenizer.texts_to_sequences(y)

# x_train과 x_test로 분할
x_train, x_test = train_test_split(padded_frames, train_size=0.8, random_state=42 , shuffle=True )

y_train, y_test = train_test_split(sequences, train_size=0.8, random_state=42 , shuffle=True )

es = EarlyStopping(monitor='val_loss' , mode='min' , patience=10 , restore_best_weights=True , verbose= 1  )

print(x_train.shape)
print(np.unique(y_train))

#2 모델
model = Sequential()
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', input_shape=(x_train[1], x_train[2], x_train[3], x_train[4])))
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu'))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(y_train[0], activation='softmax'))



#3 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train , y_train , epochs= 1 , batch_size= 1 , verbose=1 , validation_split = 0.2 )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
predict = model.predict(x_test)

print('loss =' , loss)
print('acc',accuracy_score(y_test,predict))
print('비디오 읽기 시간' , end - start)

