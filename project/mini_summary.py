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

id = 'a2'                # 모델의 이름    
mode = 'stream'          # 모델의 모드(앞의 정의한 것과 모드만 다름 base 에서 stream으로 바뀜)         
version = '3'            # 모델의 버전        
hub_url = f'https://tfhub.dev/tensorflow/movinet/{id}/{mode}/kinetics-600/classification/{version}'
model = hub.load(hub_url)       # 모델을 주소에서 가져와서 사용

model.summary()
