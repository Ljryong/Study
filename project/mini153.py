import pathlib
import tensorflow_hub as hub
import matplotlib as mpl
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import PIL
import tensorflow as tf
import tqdm
import time
import build

model_id = 'a0'
resolution = 224

# hud 에 관한 것 
id = 'a2'                       # 모델의 종류
mode = 'base'                   # 모델의 모드
version = '3'                   # 모델의 버전
hub_url = f'https://tfhub.dev/tensorflow/movinet/{id}/{mode}/kinetics-600/classification/{version}'     # 주소를 가져옴   
model = hub.load(hub_url) 


tf.keras.backend.clear_session()

# backbone = movinet.Movinet(model_id=model_id)
# backbone.trainable = False

# Set num_classes=600 to load the pre-trained weights from the original model
# model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
# model.build([None, None, None, None, 3])

# model.build([None, None, None, None, 3],num_classes = 10480 )

# Load pre-trained weights
# !wget https://storage.googleapis.com/tf_model_garden/vision/movinet/movinet_a0_base.tar.gz -O movinet_a0_base.tar.gz -q
# !tar -xvf movinet_a0_base.tar.gz

checkpoint_dir = f'movinet_{model_id}_base'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)