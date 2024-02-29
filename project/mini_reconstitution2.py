import os
import tqdm
import random
import pathlib
import imageio
import itertools
import collections
import pandas as pd


import cv2
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tensorflow_docs.vis import embed

import keras
#import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential
from keras.layers import Dense, Flatten

from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from official.projects.movinet.tools import export_saved_model

# @title Helper functions for loading data and visualizing
def list_files_per_class(zip_url):
  """
    List the files in each class of the dataset given the zip URL.

    Args:
      zip_url: URL from which the files can be unzipped.

    Return:
      files: List of files in each of the classes.
  """
#   files = []
#   with rz.RemoteZip(URL) as zip:
#     for zip_info in zip.infolist():
#       files.append(zip_info.filename)
#   return files

def get_class(fname):
  """
    Retrieve the name of the class given a filename.

    Args:
      fname: Name of the file in the UCF101 dataset.

    Return:
      Class that the file belongs to.
  """
  return fname.split('_')[-3]

def get_files_per_class(files):
  """
    Retrieve the files that belong to each class.

    Args:
      files: List of files in the dataset.

    Return:
      Dictionary of class names (key) and files (values).
  """
  files_for_class = collections.defaultdict(list)
  for fname in files:
    class_name = get_class(fname)
    files_for_class[class_name].append(fname)
  return files_for_class

def download_from_zip(zip_url, to_dir, file_names):
  """
    Download the contents of the zip file from the zip URL.

    Args:
      zip_url: Zip URL containing data.
      to_dir: Directory to download data to.
      file_names: Names of files to download.
  """
  with rz.RemoteZip(zip_url) as zip:
    for fn in tqdm.tqdm(file_names):
      class_name = get_class(fn)
      zip.extract(fn, str(to_dir / class_name))
      unzipped_file = to_dir / class_name / fn

      fn = pathlib.Path(fn).parts[-1]
      output_file = to_dir / class_name / fn
      unzipped_file.rename(output_file,)

def split_class_lists(files_for_class, count):

  split_files = []
  remainder = {}
  for cls in files_for_class:
    split_files.extend(files_for_class[cls][:count])
    remainder[cls] = files_for_class[cls][count:]
  return split_files, remainder

def download_ufc_101_subset(zip_url, num_classes, splits, download_dir):
  files = list_files_per_class(download_dir)
  for f in files:
    tokens = f.split('/')
    if len(tokens) <= 2:
      files.remove(f) # Remove that item from the list if it does not have a filename



  dirs = {}
  for split_name, split_count in splits.items():
    print(split_name, ":")
    split_dir = download_dir / split_name
    split_files, files_for_class = split_class_lists(files_for_class, split_count)
    download_from_zip(zip_url, split_dir, split_files)
    dirs[split_name] = split_dir

  return dirs

def format_frames(frame, output_size):

  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size = (172,172), frame_step = 15):

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


import shutil

class FrameGenerator:
    def __init__(self, path, n_frames, training=False):
        self.path = pathlib.Path(path)  # pathlib.Path 객체로 변환하여 경로 처리 용이하게 함
        self.n_frames = n_frames
        self.training = training
        # 클래스 이름과 ID 매핑을 생성하는 부분은 유지
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = {name: idx for idx, name in enumerate(self.class_names)}

    def get_files_and_class_names(self):
        video_paths = list(self.path.rglob('*.mp4'))  # 모든 .mp4 파일을 재귀적으로 찾음
        classes = [p.parent.name for p in video_paths]  # 비디오 파일의 부모 디렉토리 이름을 클래스 이름으로 사용
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)  # 훈련 데이터의 경우 무작위로 섞음

        for path, class_name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames)  # 비디오 프레임 추출
            label = self.class_ids_for_name[class_name]  # 클래스 이름에 해당하는 레이블 ID 추출
            yield video_frames, label

def download_hand_sign_local(local_video_dir, num_classes, splits, download_dir):
    # Ensure the download directory exists
    download_dir = pathlib.Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    # List directories (classes)
    class_dirs = [d for d in pathlib.Path(local_video_dir).iterdir() if d.is_dir()]
    random.shuffle(class_dirs)  # Shuffle to randomly select classes

    # Select a subset of class directories
    selected_class_dirs = class_dirs[:num_classes]

    # Create splits for each selected class
    dirs = {}
    for split_name, split_percentage in splits.items():
        split_dir = download_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for class_dir in selected_class_dirs:
            # List all video files in the current class directory
            video_files = list(class_dir.rglob('*.mp4'))

            # Calculate the number of files to select for the current split
            num_files = round(len(video_files) * (split_percentage / 100))
            selected_files = random.sample(video_files, num_files)

            # Create a subdirectory for the class in the split directory
            class_split_dir = split_dir / class_dir.name
            class_split_dir.mkdir(parents=True, exist_ok=True)

            # Copy selected files to the class subdirectory in the split directory
            for file_path in selected_files:
                target_path = class_split_dir / file_path.name
                shutil.copy2(file_path, target_path)

            dirs[split_name] = split_dir

    return dirs


local_video_dir = "C:\classified_video_320"
num_classes = 600  # 사용 가능한 클래스 수가 이보다 적을 수 있음
splits = {"train": 60, "val": 20, "test": 20}  # 비율을 백분율로 지정
download_dir = "C:/prodownload"

subset_paths = download_hand_sign_local(local_video_dir, num_classes, splits, download_dir)


     
batch_size = 600
num_frames = 8


# CSV 파일 경로
csv = 'C:\\Study\\new_csv_file.csv'

# CSV 파일 읽기
df = pd.read_csv(csv)

# 파일 이름으로 사용할 컬럼 이름, 예: 'video_title'
labels = df['한국어']
#CLASSES = sorted(os.listdir('./UCF101_subset/train'))

output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int32))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], num_frames, training = True),
                                          output_signature = output_signature)
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], num_frames),
                                          output_signature = output_signature)
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], num_frames),
                                         output_signature = output_signature)
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





base_model = movinet_model.MovinetClassifier(
    backbone,
    num_classes=600,
    output_states=True)

'''
x = base_model.output
x = tf.keras.layers.Dense(1024, activation='relu')(x)  # 새로운 완전 연결 레이어 추가
predictions = tf.keras.layers.Dense(419, activation='softmax')(x)  # 클래스 수를 419개로 설정

# 새로운 모델 정의
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
'''
model = movinet_model.MovinetClassifier(
    backbone,
    num_classes=600,  # 원래 클래스 수
    output_states=True)

# new_layers = [layer for layer in model.layers[:-1]]  # 마지막 층 제외하고 모든 층 가져오기
# new_output = new_layers[-1].output  # 새 마지막 층의 출력을 가져옴

# # 새 출력 층 추가 (예: 419개 클래스에 대한 분류기)
# new_output = tf.keras.layers.Dense(419, activation='softmax')(new_output)

# # 새 모델 생성
# model = tf.keras.Model(inputs=model.input, outputs=new_output)

# Create your example input here.
# Refer to the paper for recommended input shapes.
inputs = tf.ones([419, 8, 172, 172, 3])





# [Optional] Build the model and load a pretrained checkpoint.
model.build(inputs.shape)



checkpoint_dir = 'C:\study\movinet_a0_stream'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()


# Detect hardware
try:
  tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
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
  distribution_strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
  print('Running on single GPU ', gpus[0].name)
else:
  distribution_strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
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
  model = build_classifier(batch_size, num_frames, resolution, backbone, 419)
  loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
  model.compile(loss= loss_obj, optimizer=optimizer, metrics=['accuracy'])
  
  
checkpoint_path = "trained_model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)     

#=============================================================================================================#


hist = model.fit(train_ds,
                    validation_data=val_ds,batch_size=1,
                    epochs=1,
                    validation_freq=1,
                    verbose=1,
                    callbacks=[cp_callback])

model.evaluate(test_ds)#, return_dict= True)

result = model.evaluate(test_ds)

print("loss", result[0])
print("acc", result[1])


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
     
     
     
'''
def plot_confusion_matrix(actual, predicted, labels, ds_type):
  cm = tf.math.confusion_matrix(actual, predicted)
  ax = sns.heatmap(cm, annot=True, fmt='g')
  sns.set(rc={'figure.figsize':(6, 16)})
  sns.set(font_scale=1.4)
  ax.set_title('Confusion matrix of action recognition for ' + ds_type)
  ax.set_xlabel('Predicted Action')
  ax.set_ylabel('Actual Action')
  plt.xticks(rotation=90)
  plt.yticks(rotation=0)
  ax.xaxis.set_ticklabels(labels)
  ax.yaxis.set_ticklabels(labels)
  plt.show()
     

fg = FrameGenerator(subset_paths['train'], num_frames, training = True)
label_names = list(fg.class_ids_for_name.keys())
'''     
'''
aal, predicted = get_actual_predicted_labels(test_ds)
plot_confusion_matrix(actual, predicted, label_names, 'test')
'''

'''
model_id = 'a0'
use_positional_encoding = model_id in {'a3', 'a4', 'a5'}
resolution = 172

# Create backbone and model.
backbone = movinet.Movinet(
    model_id=model_id,
    causal=True,
    conv_type='2plus1d',
    se_type='2plus3d',
    activation='hard_swish',
    gating_activation='hard_sigmoid',
    use_positional_encoding=use_positional_encoding,
    use_external_states=True,
)

model = movinet_model.MovinetClassifier(
    backbone,
    num_classes=10,
    output_states=True)

# Create your example input here.
# Refer to the paper for recommended input shapes.
inputs = tf.ones([1, 13, 172, 172, 3])

# [Optional] Build the model and load a pretrained checkpoint.
model.build(inputs.shape)

# Load weights from the checkpoint to the rebuilt model
checkpoint_dir = 'trained_model'
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

def get_top_k(probs, k=5, label_map=CLASSES):
  """Outputs the top k model labels and probabilities on the given video."""
  top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
  top_labels = tf.gather(label_map, top_predictions, axis=-1)
  top_labels = [label.decode('utf8') for label in top_labels.numpy()]
  top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
  return tuple(zip(top_labels, top_probs))
     

# Create initial states for the stream model
init_states_fn = model.init_states
init_states = init_states_fn(tf.shape(tf.ones(shape=[1, 1, 172, 172, 3])))

all_logits = []

# To run on a video, pass in one frame at a time
states = init_states
for frames, label in test_ds.take(1):
  for clip in frames[0]:
    # Input shape: [1, 1, 172, 172, 3]
    clip = tf.expand_dims(tf.expand_dims(clip, axis=0), axis=0)
    logits, states = model.predict({**states, 'image': clip}, verbose=0)
    all_logits.append(logits)

logits = tf.concat(all_logits, 0)
probs = tf.nn.softmax(logits)

final_probs = probs[-1]
top_k = get_top_k(final_probs)
print()
for label, prob in top_k:
  print(label, prob)

frames, label = list(test_ds.take(1))[0]
to_gif(frames[0].numpy())

'''