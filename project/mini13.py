import tensorflow as tf
import os
import json
import pandas as pd
import re
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
import random
import requests
import json
import PIL
import joblib
from joblib import dump, load
from math import sqrt
from PIL import Image
from tqdm.auto import tqdm
import pickle
import cv2
import keras.preprocessing.text import tokenizer

# 기본 파일 위치
# BASE_PATH = 'D:/minipro//'

# with open(f'{BASE_PATH}/annotations/captions_train2017.json', 'r') as f: # 경로 안에 json 파일을 읽기모드로 열고 핸들러를 변수 f 에 할당
#     data = json.load(f)                # 핸들러 f 를 통해 josn파일을 읽음. 파이썬 딕셔너리형태로 data에 저장됨
#     data = data['annotations']         # 데이터에 'annotations'키에 해당하는 데이터만 추출해 data에 다시 덮어쓰기. caption데이터만을 담고있음

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

print(video_data)

start = time.time()

# 프레임 조정
new_width = 10
new_height = 10

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
            # 이미지 크기를 조정
            resized_frame = cv2.resize(frame, (new_width, new_height))
            frames.append(resized_frame)
            num_frames += 1
        else:
            break
    
    # 현재 비디오의 프레임 수를 저장
    video_lengths.append(num_frames)
    
    cap.release()
    
end = time.time()

print('비디오 읽기 끝')
print('비디오 읽기 시간' , end - start)

# 패딩 적용하지 않고 x 값을 생성
x_values = np.array(frames, dtype='float32')

print(x_values.shape)

MAX_LENGTH = 40
VOCABULARY_SIZE = 20000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIM = 512
UNITS = 512
EPOCHS = 30

video_keys = list(x_values.keys())   # img_to_cap_vector의 키를 불러옴 == img 파일 이름 리스트
random.shuffle(video_keys)                    # 비디오 셔플. train_test_split 에 있는 shuffle과 같은 역할

slice_index = int(len(video_keys)*0.8)        # 이미지 키 리스트에 80% 지점에 해당하는 인덱스 계산.
img_name_train_keys, img_name_val_keys = (video_keys[:slice_index],   # train_keys에 처음부터 80% 지점 직전 까지
                                          video_keys[slice_index:])   # val_keys에 80% 지점 부터 끝까지.

val_imgs = []       # 빈 리스트 생성
val_captions = []   # 빈 리스트 생성

train_dataset = tf.data.Dataset.from_tensor_slices( # 리스트나 배열같은 텐서로부터 데이터셋 생성
    (video_keys))               # train_img는 이미지파일 경로 리스트, train_captions는이미지에 해당하는 캡션의 리스트

train_dataset = train_dataset.map(                  
    load_data, num_parallel_calls=tf.data.AUTOTUNE  # load_data(코드 115번줄)로 코드 123번줄 train_dataset 전처리, numparallel_calls 는 로드와 전처리를 병렬로 처리. 성능을  최적화시켜줌
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)        # .shuffle(BUFFER_SIZE)는 전체 데이터를 BUFFER_SIZE 만큼 덩어리씩 묶은뒤 그 덩어리들을 섞는것. 덩어리 내부는 섞이지 않음
                                                        # model.fit의 batchsize 가 여기 붙은것
val_dataset = tf.data.Dataset.from_tensor_slices(   # 리스트나 배열같은 텐서로부터 데이터셋을 생성함
    (val_imgs, val_captions))                       # val_imgs 는이미지파일들의 결로를 담고 있는 리스트, val_captions는 각 이미지에 해당하는 캡션들을 담고있는 리스트

val_dataset = val_dataset.map(                      # 코드 126번줄 train_dataset과 똑같음 이름만 다름
    load_data, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

image_augmentation = tf.keras.Sequential(           # 데이터 증폭. Sequential 정의
    [   tf.keras.layers.RandomFlip("horizontal"),   # 수평방향으로 무작위로 뒤집음
        tf.keras.layers.RandomRotation(0.2),        # 최대 20% 각도로 무작위 회전 == 최대 72도
        tf.keras.layers.RandomContrast(0.3),        # 픽셀에 무작위한 대비 변화를 적용. 밝기와 명암이 조절됨
    ]
)

############### 특성 추출 #####################

def CNN_Encoder():  # InceptionV3를 이용한 CNN_Encoder 함수 정의 // 이미지의 특성을 추출하는 부분
    inception_v3 = tf.keras.applications.InceptionV3(   # keras 제공 InceptionV3. 이미지 분류를 위한 CNN 아키텍처
        include_top=False,                              # 원래 InceptionV3는 이미지 분류기 인데, 이 분류기의 최상층 레이어는 이미지 분류에만 쓰이기 때문에 불러오지 않고 특성 추출 부분까지만 불러오는 구조
        weights='imagenet'                              # ImageNet 데이터셋으로 학습된 가중치 사용. 
    )

    output = inception_v3.output                    # inception_v3를 거쳐서 나온 아웃풋을 ouput에 저장
    output = tf.keras.layers.Reshape(               # (높이,넓이,채널)형식의 3차원 형태에서 2차원 벡터로 차원 축소
        (-1, output.shape[-1]))(output)

    cnn_model = tf.keras.models.Model(inception_v3.input, output)   # 새 모델을 만들어서 인풋은 inception_v3의 인풋을, 출력은 inception_v3를 거친 출력을사용
    return cnn_model        # cnn_model은 이미지를 입력으로 받아서 InceptionV3로 이미지 특성을 추출하는 모델

class TransformerEncoderLayer(tf.keras.layers.Layer):   # 트랜스포머의 인코더레이어

    def __init__(self, embed_dim, num_heads):       # embed_dim = 임베딩 차원, num_heads는 어텐션 헤드의 갯수
        super().__init__()              # __init__메서트에서 클래스 초기화
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()    # 입력을 정규화하는 레이어 생성. 트랜스포머에서는 레이어 정규화가 중요한 역할을함
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()    # 
        self.attention = tf.keras.layers.MultiHeadAttention(        # 다중헤드어텐션 연산 수행
            num_heads=num_heads, key_dim=embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu") # 입력을 임베팅차원에 대비해서 확장시키기 위해 Dense레이어를 하나 추가함. embed_dim 만큼 노드(=차원)이 늘어남
    

    def call(self, x, training):    # call에서 실제로 입력데이터를 처리함. x가 데이터, training은 현재 학습중인지 묻는것
        x = self.layer_norm_1(x)    # 입력 데이터를 첫번째 레이어 정규화에 통과시킴
        x = self.dense(x)           # 처리된 입력을 임베딩 차원에 대비해서 확장시키는 Dense레이어에 통과시킴

        attn_output = self.attention(   # 멀티헤드 어텐션 연산 수행. 어텐션 출력 attn_ouput에 저장
            query=x,
            value=x,
            key=x,
            attention_mask=None,
            training=training       # 훈련인지 테스트인지 나타내고 dropout같은 정규화 기법이 적용될지를 결정함. test모드라면 적용안됨
        )

        x = self.layer_norm_2(x + attn_output)  # 입력과 어텐션출력을 더하고 두번째 레이어 정규화 실행
        return x            # residual connection이라고도 하며 입력과 출력간의 차이를 줄여줌. 입력과 어텐션 출력을결합한 후 정보를 안정화시키고 학습을 도와줌

class Embeddings(tf.keras.layers.Layer):        # 트랜스포머 모델의 입력으로 사용될 토큰 및 위치 임베딩을 생성
# 임베딩이란 : 이미지의 저차원적 특성 벡터를 추출해 유사도가 높은 단어끼리는 임베딩 공간상에서 서로 가까운 곳에 위치하게 됨. 즉 유사성을 띈 단어들 간의 분류를 위함.
    def __init__(self, vocab_size, embed_dim, max_len): 
        super().__init__()      # tf.keras.layers.Layer 를 상속받은 Embeddings 클래스이기 때문에 tf.keras.layers.Layer의 생성자를 호출. 부모 클래스의 모든 속성과 메서드를 상속받게됨
        self.token_embeddings = tf.keras.layers.Embedding(  # 단어 집합의 크기(vocab_size)와 임베딩 차원(embed_dim)을 인자로 받아 각 단어를 고정된 길이의 밀집 벡터로 임베딩
            vocab_size, embed_dim)              # 단어의 의미보존, 차원축소, 단어간 관계표현 
        self.position_embeddings = tf.keras.layers.Embedding(   # 위치 임베딩. 입력 시퀀스의 위치정보를 표현하기위해 위치 임베딩을 사용.
            max_len, embed_dim, input_shape=(None, max_len))    # 입력 시퀀스 안에서 각 토큰의 상대적인 위치를 나타내는임베딩임
    

    def call(self, input_ids):
        length = tf.shape(input_ids)[-1]    # input_ids 텐서의 마지막 차원의 길이를 구함. 입력 시퀀스의 길이를 나타냄
        position_ids = tf.range(start=0, limit=length, delta=1) # length만큼의 길이를 가지고 0부터 1씩 증가하는 숫자배열 생성. / 입력 시퀀스의 위치정보를 나타내는 벡터
        position_ids = tf.expand_dims(position_ids, axis=0) # position_ids텐서에 차원을 추가해 (1,length)로 변경함 / 임베딩에 적절한 벡터형태로 변환

        token_embeddings = self.token_embeddings(input_ids) # 입력으로 받은 토큰 시퀀스에 대한 토큰 임베딩 계산. 입력 시퀀스(문장) 내 각 토큰에 대한 단어 임베딩임.
        position_embeddings = self.position_embeddings(position_ids)    # 입력 시퀀스내 각 토큰의 순서를 나타내는 임베딩. 각 토큰(단어)의 위치 정보에 대한 임베딩임.

        return token_embeddings + position_embeddings   # 토큰 임베딩과 위치 임베딩을더해 최종 임베딩 생성 및 반환. 
    # 이제 입력 시퀀스의 각 토큰에 대한 토큰임베딩과 위치임베딩을 결합한 결과를 얻을수 있음 / 여기서 더하는건 실제 덧셈 연산을 진행하는것
class TransformerDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, units, num_heads):
        super().__init__()      # tf.keras.layers.Layer클래스 생성자 정의. 여러 구성층 초기화.
        self.embedding = Embeddings(        # 
            tokenizer.vocabulary_size(), embed_dim, MAX_LENGTH)

        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )

        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()

        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)

        self.out = tf.keras.layers.Dense(tokenizer.vocabulary_size(), activation="softmax")

        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
    

    def call(self, input_ids, encoder_output, training, mask=None):
        embeddings = self.embedding(input_ids)

        combined_mask = None
        padding_mask = None
        
        if mask is not None:
            causal_mask = self.get_causal_attention_mask(embeddings)
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        attn_output_1 = self.attention_1(
            query=embeddings,
            value=embeddings,
            key=embeddings,
            attention_mask=combined_mask,
            training=training
        )

        out_1 = self.layernorm_1(embeddings + attn_output_1)

        attn_output_2 = self.attention_2(
            query=out_1,
            value=encoder_output,
            key=encoder_output,
            attention_mask=padding_mask,
            training=training
        )

        out_2 = self.layernorm_2(out_1 + attn_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds


    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )
        return tf.tile(mask, mult)

class ImageCaptioningModel(tf.keras.Model):

    def __init__(self, cnn_model, encoder, decoder, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.image_aug = image_aug
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")


    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)
    

    def compute_loss_and_acc(self, img_embed, captions, training=True):
        encoder_output = self.encoder(img_embed, training=True)
        y_input = captions[:, :-1]
        y_true = captions[:, 1:]
        mask = (y_true != 0)
        y_pred = self.decoder(
            y_input, encoder_output, training=True, mask=mask
        )
        loss = self.calculate_loss(y_true, y_pred, mask)
        acc = self.calculate_accuracy(y_true, y_pred, mask)
        return loss, acc

    
    def train_step(self, batch):
        imgs, captions = batch

        if self.image_aug:
            imgs = self.image_aug(imgs)
        
        img_embed = self.cnn_model(imgs)

        with tf.GradientTape() as tape:
            loss, acc = self.compute_loss_and_acc(
                img_embed, captions
            )
    
        train_vars = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}
    

    def test_step(self, batch):
        imgs, captions = batch

        img_embed = self.cnn_model(imgs)

        loss, acc = self.compute_loss_and_acc(
            img_embed, captions, training=False
        )

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)
decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8)

cnn_model = CNN_Encoder()
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation,
)

cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="none"
)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor = 'val_acc')

caption_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=cross_entropy
)

history = caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stopping]
)

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

def load_image_from_path(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def generate_caption(img_path, add_noise=False):
    img = load_image_from_path(img_path)
    
    if add_noise:
        noise = tf.random.normal(img.shape)*0.1
        img = img + noise
        img = (img - tf.reduce_min(img))/(tf.reduce_max(img) - tf.reduce_min(img))
    
    img = tf.expand_dims(img, axis=0)
    img_embed = caption_model.cnn_model(img)
    img_encoded = caption_model.encoder(img_embed, training=False)

    y_inp = '[start]'
    for i in range(MAX_LENGTH-1):
        tokenized = tokenizer([y_inp])[:, :-1]
        mask = tf.cast(tokenized != 0, tf.int32)
        pred = caption_model.decoder(
            tokenized, img_encoded, training=False, mask=mask)
        
        pred_idx = np.argmax(pred[0, i, :])
        pred_idx = tf.convert_to_tensor(pred_idx)
        pred_word = idx2word(pred_idx).numpy().decode('utf-8')
        if pred_word == '[end]':
            break
        
        y_inp += ' ' + pred_word
    
    y_inp = y_inp.replace('[start] ', '')
    return y_inp

idx = random.randrange(0, len(captions))
img_path = captions.iloc[idx].image

pred_caption = generate_caption(img_path)
print('Predicted Caption:', pred_caption)
print()
Image.open(img_path)

img_url = "https://images.squarespace-cdn.com/content/v1/5e0e65adcd39ed279a0402fd/1627422658456-7QKPXTNQ34W2OMBTESCJ/1.jpg?format=2500w"

im = Image.open(requests.get(img_url, stream=True).raw)
im = im.convert('RGB')
im.save('tmp.jpg')

pred_caption = generate_caption('tmp.jpg', add_noise=False)
print('Predicted Caption:', pred_caption)
print()
im.show()

# 가중치 저장
caption_model.save_weights('c:/Study/project/group_project/min/save/caption_model.h5')
# pickle.dump(caption_model, open('c:/Study/project/group_project/min/caption_model.dat', 'wb'))    # error
# pickle.dump(caption_model, open('c:/Study/project/group_project/min/caption_model.pkl', 'wb'))

# dump(caption_model, 'c:/Study/project/group_project/min/save/caption_model.joblib')