import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from efficientnet.keras import EfficientNetB0
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import xgboost
from sklearn.model_selection import train_test_split

# 하이퍼파라미터 설정
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 41
EPOCHS = 5

# 시드 설정
np.random.seed(SEED)

# 데이터 경로
path = 'C:/_data/dacon/bird'

# 데이터 불러오기
df = pd.read_csv(os.path.join(path, 'train.csv'))

# 데이터 전처리
le = preprocessing.LabelEncoder()
df['label'] = le.fit_transform(df['label'])
train, val = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=SEED)

# 데이터 제너레이터 생성
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3,
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory=path,
    x_col='img_path',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='raw',
    subset='training',
    seed=SEED
)
x_train , y_train = train_test_split(train_generator , train_size= 0.8 , random_state= 0.2  , )

val_generator = train_datagen.flow_from_dataframe(
    dataframe=val,
    directory=path,
    x_col='img_path',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='raw',
    subset='validation',
    seed=SEED
)

# 모델 정의
""" base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')  # 클래스 수에 맞게 조정
])
"""
model = xgboost.XGBClassifier(randomstate = 42)

# 모델 컴파일
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
history = model.fit(
    train_generator,
    # steps_per_epoch=train_generator.samples // BATCH_SIZE,
    # epochs=EPOCHS,
    # validation_data=val_generator,
    # validation_steps=val_generator.samples // BATCH_SIZE
)

# 예측 및 평가
test_df = pd.read_csv(os.path.join(path, 'test.csv'))
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=path,
    x_col='img_path',
    y_col=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
)
preds = model.predict(test_generator)
pred_labels = np.argmax(preds, axis=1)
pred_labels = le.inverse_transform(pred_labels)

# 결과 저장
submit = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
submit['label'] = pred_labels
submit.to_csv(os.path.join(path, 'baseline_submit.csv'), index=False)
