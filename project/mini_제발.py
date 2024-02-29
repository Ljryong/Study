import os
import tqdm
import random
import pathlib
import imageio
import itertools
import collections

import cv2
import numpy as np
import pandas as pd
import remotezip as rz
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tensorflow_docs.vis import embed

import keras
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from official.projects.movinet.tools import export_saved_model

# 데이터 경로 설정
video_path = 'D:/minipro//'
csv_path = 'C:/Study//'

# 디렉토리 내 파일 목록 가져오기
file_list = os.listdir(video_path)
labels_df = pd.read_csv(csv_path + 'new_csv_file.csv')

# 영상 파일 이름을 기준으로 CSV 파일과 매핑
video_data = []
for file_name in file_list:
    # CSV 파일에서 해당 파일 이름에 해당하는 데이터 가져오기
    video_info = labels_df[labels_df['파일명'] == file_name]
    if not video_info.empty:
        # 필요한 정보 가져오기 예시: 레이블 가져오기
        label = video_info['한국어'].values[0]
        # 영상 파일의 경로와 함께 데이터 저장
        video_data.append((os.path.join(video_path, file_name), label))

# 영상 데이터와 레이블 매핑 결과 출력
for video_path, label in video_data:
    print(f"영상 파일: {video_path}, 레이블: {label}")

print(video_data)
print('매핑 끝')

# 레이블 데이터 추출
labels = labels_df['한국어']

print(labels)

# 영상 데이터 처리 함수
def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
        frame: Image that needs to resized and padded.
        output_size: Pixel size of the output frame image.

    Return:
        Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def frames_from_video_file(video_path, n_frames, output_size=(172, 172), frame_step=15):
    """
    Creates frames from each video file present for each category.

    Args:
        video_path: File path to the video.
        n_frames: Number of frames to be created per video file.
        output_size: Pixel size of the output frame image.

    Return:
        An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result

def to_gif(images):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./animation.gif', converted_images, fps=10)
    return embed.embed_file('./animation.gif')


class FrameGenerator:
    def __init__(self, train_dir, val_dir, test_dir, n_frames, training=False):
        """ Returns a set of frames with their associated label.

          Args:
            train_dir: Directory containing training video files.
            val_dir: Directory containing validation video files.
            test_dir: Directory containing test video files.
            n_frames: Number of frames.
            training: Boolean to determine if training dataset is being created.
        """
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.n_frames = n_frames
        self.training = training

        # Get class names from the train directory
        self.class_names = sorted(os.listdir(train_dir))

        # Dictionary to map class names to IDs
        self.class_ids_for_name = {name: idx for idx, name in enumerate(self.class_names)}

    def get_files_and_class_names(self, directory):
        video_paths = list(pathlib.Path(directory).glob('*/*.avi'))
        classes = [p.parent.name for p in video_paths]
        return video_paths, classes

    def __call__(self):
        if self.training:
            video_paths, classes = self.get_files_and_class_names(self.train_dir)
        else:
            video_paths, classes = self.get_files_and_class_names(self.test_dir if self.test_dir else self.val_dir)

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames)
            label = self.class_ids_for_name[name]  # Encode labels
            yield video_frames, label



# Helper functions below used are taken from following tutorials
# https://www.tensorflow.org/tutorials/video/video_classification
# https://www.tensorflow.org/tutorials/video/transfer_learning_with_movinet

# URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'
# download_dir = pathlib.Path('./UCF101_subset/')
# subset_paths = download_ufc_101_subset(URL,
#                         num_classes = 10,
#                         splits = {"train": 40, "val": 10, "test": 10},
#                         download_dir = download_dir)

import shutil


def download_ufc_101_subset_from_local(local_video_dir, num_classes, splits, download_dir):
    """
    Load a subset of the UFC101 dataset from a local directory and split them into various parts, such as
    training, validation, and test.

    Args:
      local_video_dir: Local directory path containing video data.
      num_classes: Number of labels/classes to include.
      splits: Dictionary specifying the division of data into training, validation, test, etc. 
              (key) with the number of files per split (value).
      download_dir: Directory to save the split data to.

    Return:
      dirs: Dictionary with keys being the split names and values being the paths to the directories containing the data.
    """
    # Ensure the download directory exists
    download_dir = pathlib.Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    # List all video files
    video_files = [file for file in os.listdir(local_video_dir) if file.endswith(('.mp4', '.avi'))]
    random.shuffle(video_files)  # Shuffle to randomly select for splits

    # Assuming video_files are named in a way that includes class information, e.g., class_video01.mp4
    # Extract unique classes
    classes = sorted(set(file.split('_')[0] for file in video_files))[:num_classes]

    # Filter videos by selected classes
    selected_videos = [file for file in video_files if file.split('_')[0] in classes]

    # Create splits
    dirs = {}
    remaining_videos = selected_videos
    for split_name, num_files in splits.items():
        split_dir = download_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Select a subset of files for the current split
        selected_for_split = remaining_videos[:num_files]
        remaining_videos = remaining_videos[num_files:]

        # Copy selected files to the split directory
        for file in selected_for_split:
            shutil.copy2(os.path.join(local_video_dir, file), split_dir)

        dirs[split_name] = split_dir

    return dirs


# Example usage:
local_video_dir = 'D:\\minipro\\'
num_classes = 600
splits = {"train": 40, "val": 10, "test": 10}
download_dir = 'D:\\minipro\\'

subset_paths = download_ufc_101_subset_from_local(local_video_dir, num_classes, splits, download_dir)

batch_size = 5
num_frames = 8

CLASSES = sorted(os.listdir('D:/minipro/train'))

output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int32))

# 데이터 경로 설정
# train_dir = 'D:/minipro/train'
# val_dir = 'D:/minipro/val'
# test_dir = 'D:/minipro/test'

# train_ds = tf.data.Dataset.from_generator(FrameGenerator(pathlib.Path('train'), num_frames, training = True),
#                                           output_signature = output_signature)
# train_ds = train_ds.batch(batch_size)

# val_ds = tf.data.Dataset.from_generator(FrameGenerator(pathlib.Path('val'), num_frames),
#                                           output_signature = output_signature)
# val_ds = val_ds.batch(batch_size)

# test_ds = tf.data.Dataset.from_generator(FrameGenerator(pathlib.Path('test'), num_frames),
#                                          output_signature = output_signature)
# test_ds = test_ds.batch(batch_size)

# gpt
train_dir = 'D:/minipro/train'
val_dir = 'D:/minipro/val'
test_dir = 'D:/minipro/test'

train_ds = tf.data.Dataset.from_generator(FrameGenerator(train_dir, val_dir, test_dir, num_frames, training=True),
                                          output_signature=output_signature)
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(train_dir, val_dir, test_dir, num_frames),
                                        output_signature=output_signature)
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(train_dir, val_dir, test_dir, num_frames),
                                         output_signature=output_signature)
test_ds = test_ds.batch(batch_size)


for frames, labels in train_ds.take(1):
    print(f"Shape: {frames.shape}")
    print(f"Label: {labels.shape}")

model_id = 'a0'
use_positional_encoding = model_id in {'a1', 'a2', 'a3', 'a4', 'a5'}
resolution = 172

backbone = movinet.Movinet(
    model_id=model_id,
    causal=True,
    conv_type='2plus1d',
    se_type='2plus3d',
    activation='hard_swish',
    gating_activation='hard_sigmoid',
    use_positional_encoding=use_positional_encoding,
    use_external_states=False,
)

# Note: this is a temporary model constructed for the
# purpose of loading the pre-trained checkpoint. Only
# the backbone will be used to build the custom classifier.

model = movinet_model.MovinetClassifier(
    backbone,
    num_classes=600,
    output_states=True)

# Create your example input here.
# Refer to the paper for recommended input shapes.
inputs = tf.ones([10480, 10, 172, 172, 3])

# [Optional] Build the model and load a pretrained checkpoint.
model.build(inputs.shape)

'''
url에서 가중치 불러오기
# Extract pretrained weights
!wget https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a0_stream.tar.gz -O movinet_a0_stream.tar.gz -q
!tar -xvf movinet_a0_stream.tar.gz
'''

# checkpoint_dir = 'C:\study\movinet_a0_stream'
# checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
# checkpoint = tf.train.Checkpoint(model=model)
# status = checkpoint.restore(checkpoint_path)
# status.assert_existing_objects_matched()

# Restore the model from the checkpoint
checkpoint_dir = 'D:/movinet_a0_stream'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

# Detect hardware
try:
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
except ValueError:
    tpu_resolver = None
    gpus = tf.config.experimental.list_logical_devices("GPU")

# Select appropriate distribution strategy
if tpu_resolver:
    tf.config.experimental_connect_to_cluster(tpu_resolver)
    tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
    distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu_resolver)
    print('Running on TPU ', tpu_resolver.cluster_spec().as_dict()['worker'])
elif len(gpus) > 1:
    distribution_strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
elif len(gpus) == 1:
    distribution_strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
    print('Running on single GPU ', gpus[0].name)
else:
    distribution_strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
    print('Running on CPU')

print("Number of accelerators: ", distribution_strategy.num_replicas_in_sync)


def build_classifier(batch_size, num_frames, resolution, backbone, num_classes):
    """Builds a classifier on top of a backbone model."""
    model = movinet_model.MovinetClassifier(
        backbone=backbone,
        num_classes=num_classes)
    model.build([batch_size, num_frames, resolution, resolution, 3])

    return model


# Construct loss, optimizer and compile the model
with distribution_strategy.scope():
    model = build_classifier(batch_size, num_frames, resolution, backbone, 10)
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

checkpoint_path = "trained_model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

tf.config.run_functions_eagerly(True)

model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'], run_eagerly=True)

hist = model.fit(train_ds,
                 validation_data=val_ds,
                 epochs=1,
                 validation_freq=1,
                 verbose=1,
                 callbacks=[cp_callback])

model.evaluate(test_ds)#, return_dict= True)

result = model.evaluate(test_ds)

print("제발..", result)

def get_actual_predicted_labels(dataset):
    """
      Create a list of actual ground truth values and the predictions from the model.

      Args:
        dataset: An iterable data structure, such as a TensorFlow Dataset, with features and labels.

      Return:
        Ground truth and predicted values for a particular dataset.
    """
    actual = [labels for _, labels in dataset.unbatch()]
    predicted = model.predict(dataset)

    actual = tf.stack(actual, axis=0)
    predicted = tf.concat(predicted, axis=0)
    predicted = tf.argmax(predicted, axis=1)

    return actual, predicted

# print('영상', ) 
print('라벨', labels)
