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
import os
import cv2
import pandas as pd

st_t = time.time()

# 그래프를 보여줄 때의 폰트를 10으로 지정
mpl.rcParams.update({
    'font.size': 10,
})

# 수화 영상을 가져오는 것 x값을 바꾸려면 여길 만져야 됨
start = time.time()

video_path = 'D:/minipro//'

# 디렉토리 내 파일 목록 가져오기
file_list = os.listdir(video_path)

# 프레임 조정
new_width = 100
new_height = 100

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

# 점핑잭 영상의 라벨을 가져오는 것 y값을 바꾸려면 여길 만져야 됨
csv_path = 'C:/Study//'
y = pd.read_csv(csv_path + 'new_csv_file.csv')
y_labels = y['한국어']

print(y_labels.shape)

# hub 에 관한 것 
id = 'a2'                       # 모델의 이름
mode = 'base'                   # 모델의 모드
version = '3'                   # 모델의 버전
hub_url = f'https://tfhub.dev/tensorflow/movinet/{id}/{mode}/kinetics-600/classification/{version}'     # 주소를 가져옴   
model = hub.load(hub_url)       # 모델을 주소에서 가져와서 사용

# 모델의 서명을 가져옴, 없어도 되는데 없을 경우 가끔 문제가 발생하여서 있는게 좋음
sig = model.signatures['serving_default']
print(sig.pretty_printed_signature())

#warmup
sig(image = x_values[tf.newaxis, :1])              # 서명에 입력 이미지를 전달한다, 이미지에 차원을 추가하여 형식에 맞게 변환 시켜줌 // 1차원을 2차원으로 만들어줌

logits = sig(image = x_values[tf.newaxis, ...])    # 서명에 전체 입력 이미지를 전달한다, 이미지에 차원을 추가하여 형식에 맞게 변환 시켜줌 
logits = logits['classifier_head'][0]                 # 첫번째 예측 값만 가져온다 

print(logits.shape)

#@title
# Get top_k labels and probabilities
def get_top_k(probs, k=5, label_map=y_labels):
  """Outputs the top k model labels and probabilities on the given video.

  Args:
    probs: probability tensor of shape (num_frames, num_classes) that represents
      the probability of each class on each frame.
    k: the number of top predictions to select.
    label_map: a list of labels to map logit indices to label strings.

  Returns:
    a tuple of the top-k labels and probabilities.
  """
  # Sort predictions to find top_k
  top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
  # probs: 각 프레임 및 각 클래스에 대한 확률을 나타내는 (프레임 수, 클래스 수) 형태의 확률 텐서입니다
  # k: 상위 K개의 예측의 라벨을 수집
  # axis=-1을 사용하면 주어진 텐서의 마지막 차원을 기준으로 연산을 수행합니다.
  # top_predictions는 확률 텐서인 probs를 내림차순으로 정렬한 후 상위 k개의 요소의 색인을 나타냅니다. 
  # 이것들은 가장 높은 확률을 가진 클래스의 색인입니다. 이를 통해 가장 확률이 높은 상위 k개의 클래스를 예측할 수 있습니다.
  # 확률 텐서 : 입력 데이터가 각 클래스에 속할 확률을 나타냅니다
  top_labels = tf.gather(label_map, top_predictions, axis=-1)
  # label_map: 로짓 인덱스를 라벨 문자열로 매핑하기 위한 라벨 목록입니다.
  # 확률 텐서인 top_predictions을 써서 label_map에서 각 클래스에 해당하는 라벨을 가져오기
  # 가져온 라벨들을 문자열(bytes)형식으로 되어 있음
  top_labels = [label.decode('utf8') for label in top_labels.numpy()]
  # bytes의 형식인 문자열을 utf8 형식을 바꾼다
  top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
  # gather 함수를 사용하여 probs 텐서에서 top_predictions에 해당하는 인덱스의 확률으 가져온다
  # 결과로 얻어지는 top_probs는 상위 K개의 확률 값을 담은 NumPy 배열이 됩니다.
  return tuple(zip(top_labels, top_probs))
  # top_labels, top_probs 을 튜플 형태로 바꿔서 묶어서(zip) 반환한다. (첫번째 라벨, 첫번째 확률) 이런 식으로 나옴
probs = tf.nn.softmax(logits, axis=-1)
# softmax 함수를 사용하여 각각의 확률 값을 변환하여 반환 , axis=-1 은 마지막 클래스의 대한 축?을 기준으로 적용한다
for label, p in get_top_k(probs):
  print(f'{label:20s}: {p:.3f}')
# Label_1            : 0.750
# Label_2            : 0.220
# Label_3            : 0.015
# Label_4            : 0.010
# Label_5            : 0.005
# for 문을 이용하여 각각의 라벨과 확률을 보여준다

id = 'a2'                # 모델의 이름    
mode = 'stream'          # 모델의 모드(앞의 정의한 것과 모드만 다름 base 에서 stream으로 바뀜)         
version = '3'            # 모델의 버전        
hub_url = f'https://tfhub.dev/tensorflow/movinet/{id}/{mode}/kinetics-600/classification/{version}'
model = hub.load(hub_url)       # 모델을 주소에서 가져와서 사용

list(model.signatures.keys())   # 모델이 가지고 있는 서명들의 키(key)를 리스트로 반환합니다. 이는 모델이 지원하는 서명들의 목록을 확인할 수 있습니다

lines = model.signatures['init_states'].pretty_printed_signature().splitlines()
# 모델에서 'init_states' 서명에 대한 정보를 출력합니다.
# pretty_printed_signature() 함수는 해당 서명의 서명 정보를 문자열 형태로 반환합니다.
# 반환된 문자열을 줄 단위로 분할하여 리스트에 저장합니다
lines = lines[:10]        # 너무 많은 정보를 넣을까봐 10개로 자른 것
lines.append('      ...') # 출력이 10줄을 넘어갔을 경우 정보가 더 있다는 것을 보여주기 위해 ...을 넣어준다
print('.\n'.join(lines))  # 줄 단위로 분할된 정보들을 다시 하나의 문자열로 결합하여 출력 , 줄이 바뀌는 것은 . 으로 표시된다


initial_state = model.init_states(x_values[tf.newaxis, ...].shape)
# 모델의 초기 상태를 초기화합니다. 이 초기 상태는 입력 데이터의 형태(shape)에 따라 생성됩니다.
# shape 는 jumpingjack 비디오 데이터의 형태에 새로운 차원을 추가하여 형태(shape)를 얻습니다.
# 이 형태 정보를 사용하여 모델의 초기 상태를 초기화
type(initial_state)
# initial_state 변수의 데이터 타입을 출력합니다. 초기 상태는 일반적으로 딕셔너리(dictionary) 형태로 표현됩니다

list(sorted(initial_state.keys()))[:5]
# 초기 상태 딕셔너리의 키(keys)를 정렬하여 처음 5개를 리스트로 출력합니다.
# 이는 초기 상태에 포함된 정보 중 일부를 확인하기 위한 것입니다.

inputs = initial_state.copy()
# 초기 상태를 복사하여 inputs 변수에 저장합니다.
# 일반적으로 모델에 입력할 데이터를 준비할 때, 초기 상태를 복사하여 사용하는 것이 일반적입니다.


# Add the batch axis, take the first frme, but keep the frame-axis.
inputs['image'] = x_values[tf.newaxis, 0:1, ...] 
#첫번째 프레임만 입력,          ^차원을 추가해 모델이 원하는 입력형태로 만들어줌.
     

# warmup
model(inputs)
#실제 예측 전 모델의 성능을 향상시키기 위해 첫번째 프레임만 넣고 웜업     
     

logits, new_state = model(inputs) #모델에 x를 통과해 예측결과를 냄/ 모델의 새로운 상태를 얻어냄.
logits = logits[0]
probs = tf.nn.softmax(logits, axis=-1) #클래스에 대한 확률 분포

for label, p in get_top_k(probs):  # 확률이 가장 높은 상위 k개 라벨&확률 출력.
  print(f'{label:20s}: {p:.3f}')

#print()


state = initial_state.copy()              #동영상의 각 프레임에 위의 과정 반복.
all_logits = []                           #프레임별로 모델의 예측결과 얻기 가능.

for n in range(len(x_values)):
  inputs = state
  inputs['image'] = x_values[tf.newaxis, n:n+1, ...]
  result, state = model(inputs)
  all_logits.append(logits)

probabilities = tf.nn.softmax(all_logits, axis=-1) # 모든 프레임에 대한 예측결과를 softmax에 통과, 프레임별 클래스 확률분포 계산.
     

#결과출력
for label, p in get_top_k(probabilities[-1]):
  print(f'{label:20s}: {p:.3f}')
     

id = tf.argmax(probabilities[-1])
plt.plot(probabilities[:, id]) #동작의 확률이 어떻게 변하는지 시각화하여줌
plt.xlabel('Frame #')
plt.ylabel(f"p('{y_labels[id]}')")




for label, p in get_top_k(tf.reduce_mean(probabilities, axis=0)):
  print(f'{label:20s}: {p:.3f}')
  
#@title
# Get top_k labels and probabilities predicted using MoViNets streaming model


#상위 k개의 라벨을 반환해주는 클래스 만들기
# probs- 각 프레임에 대한 클래스의 확률을 나타내는 텐서(num_frame(비디오프레임수), num_classes(분류할 클래스수))형태
# label_map = 문자열 매핍에 사용되는 라벨 리스트
def get_top_k_streaming_labels(probs, k=5, label_map=y_labels):
  
  
  """Returns the top-k labels over an entire video sequence.

  Args:
    probs: probability tensor of shape (num_frames, num_classes) that represents
      the probability of each class on each frame.
    k: the number of top predictions to select.
    label_map: a list of labels to map logit indices to label strings.

  Returns:
    a tuple of the top-k probabilities, labels, and logit indices
  """
  #top_categories_last - 마지막 프레임에서 가장 확률이 높은 카테고리 인덱스 반환  
  top_categories_last = tf.argsort(probs, -1, 'DESCENDING')[-1, :1]
  
  #각 프레임에 대한 상위 k개 예측
  categories = tf.argsort(probs, -1, 'DESCENDING')[:, :k]
  
  #k개 인덱스를 1차원 배열로 반환
  categories = tf.reshape(categories, [-1])


  counts = sorted([
      #sorted로 튜플리스트를 내림차순으로 정렬.
      #리스트에서 반복 수행. 각 카테고리 인덱스. 나타내는 총 횟수를 튜플로 생성
      (i.numpy(), tf.reduce_sum(tf.cast(categories == i, tf.int32)).numpy())
      
      #상위 k개의 예측된 카테고리 식별
      for i in tf.unique(categories)[0]
  ], key=lambda x: x[1], reverse=True)

  #top_probs_idx = 마지막 프레임에서 가장 확률이 높은 카테고리 인덱스가 있음.
  top_probs_idx = tf.constant([i for i, _ in counts[:k]])
  
  #인덱스와 카테고리를 엮음
  top_probs_idx = tf.concat([top_categories_last, top_probs_idx], 0)
  #중복 제거, 고유 인덱스 선택. +1은 마지막 프레임의 확률 카테고리를 포함   
  top_probs_idx = tf.unique(top_probs_idx)[0][:k+1]
  # 각 카테고리에 대한 확률 수집
  top_probs = tf.gather(probs, top_probs_idx, axis=-1)
  # 차원을 전환해 확률이 바르게 정렬되게 만듬.
  top_probs = tf.transpose(top_probs, perm=(1, 0))
  # 각 라벨에 대한 확률 수집
  top_labels = tf.gather(label_map, top_probs_idx, axis=0)
  # 읽을수 있는 문자열로 다시 디코딩.
  top_labels = [label.decode('utf8') for label in top_labels.numpy()]

  #k개 확률. 해당 라벨, 인덱스를 반환
  return top_probs, top_labels, top_probs_idx






# k개 예측을 시각화 하는데 사용.
# Plot top_k predictions at a given time step
def plot_streaming_top_preds_at_step(
    top_probs,                           
    top_labels,
    step=None,
    image=None,
    legend_loc='lower left',
    duration_seconds=10,
    figure_height=500,
    playhead_scale=0.8,
    grid_alpha=0.3):
  """Generates a plot of the top video model predictions at a given time step.

  Args:
    top_probs: a tensor of shape (k, num_frames) representing the top-k
      probabilities over all frames.
    top_labels: a list of length k that represents the top-k label strings.
    step: the current time step in the range [0, num_frames].
    image: the image frame to display at the current time step.
    legend_loc: the placement location of the legend.
    duration_seconds: the total duration of the video.
    figure_height: the output figure height.
    playhead_scale: scale value for the playhead.
    grid_alpha: alpha value for the gridlines.

  Returns:
    A tuple of the output numpy image, figure, and axes.
  """
  '''
  입력 인자
top_probs: 모든 프레임에 대한 상위 k개 확률을 나타내는 (k, num_frames) 형태의 텐서입니다.
top_labels: 상위 k개 라벨 문자열을 나타내는 길이 k의 리스트입니다.
step: [0, num_frames] 범위의 현재 시간 단계입니다.
image: 현재 시간 단계에서 표시할 이미지 프레임입니다.
legend_loc: 범례의 위치를 나타냅니다.
duration_seconds: 비디오의 총 기간(초)입니다.
figure_height: 출력 그림의 높이입니다.
playhead_scale: 플레이헤드의 스케일 값을 설정합니다.
grid_alpha: 그리드 라인의 알파 값입니다. (그래프 라인의 투명도)
'''
  
  
  # find number of top_k labels and frames in the video
  #k개 라벨, 프레임 수
  num_labels, num_frames = top_probs.shape
  
  
  
  #그래프 구성하기
  # 마지막 프레임을 기본값으로 설정.
  if step is None:
    step = num_frames
  # Visualize frames and top_k probabilities of streaming video
  fig = plt.figure(figsize=(6.5, 7), dpi=300)
  gs = mpl.gridspec.GridSpec(8, 1)
  ax2 = plt.subplot(gs[:-3, :])
  ax = plt.subplot(gs[-3:, :])
  # display the frame
  if image is not None:
    #이미지를 상단에 표시
    ax2.imshow(image, interpolation='nearest')
    ax2.axis('off')
    
  # 확률 그래프 그리기  
  # x-axis (frame number)
  preview_line_x = tf.linspace(0., duration_seconds, num_frames)
  # y-axis (top_k probabilities)
  preview_line_y = top_probs

  line_x = preview_line_x[:step+1]
  line_y = preview_line_y[:, :step+1]

  for i in range(num_labels):
    ax.plot(preview_line_x, preview_line_y[i], label=None, linewidth='1.5',
            linestyle=':', color='gray')
    ax.plot(line_x, line_y[i], label=top_labels[i], linewidth='2.0')


  ax.grid(which='major', linestyle=':', linewidth='1.0', alpha=grid_alpha)
  ax.grid(which='minor', linestyle=':', linewidth='0.5', alpha=grid_alpha)

  min_height = tf.reduce_min(top_probs) * playhead_scale
  max_height = tf.reduce_max(top_probs)
  ax.vlines(preview_line_x[step], min_height, max_height, colors='red')
  ax.scatter(preview_line_x[step], max_height, color='red')

  ax.legend(loc=legend_loc)

  plt.xlim(0, duration_seconds)
  plt.ylabel('Probability')
  plt.xlabel('Time (s)')
  plt.yscale('log')

  fig.tight_layout()
  fig.canvas.draw()

  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()

  figure_width = int(figure_height * data.shape[1] / data.shape[0])
  image = PIL.Image.fromarray(data).resize([figure_width, figure_height])
  image = np.array(image)

  return image




# Plotting top_k predictions from MoViNets streaming model
def plot_streaming_top_preds(
    probs,
    video,                #표시할 비디오 데이터
    top_k=5,
    video_fps=25.,
    figure_height=500,
    use_progbar=True):
  """Generates a video plot of the top video model predictions.

  Args:
    probs: probability tensor of shape (num_frames, num_classes) that represents
      the probability of each class on each frame.
    video: the video to display in the plot.
    top_k: the number of top predictions to select.
    video_fps: the input video fps.
    figure_fps: the output video fps.
    figure_height: the height of the output video.
    use_progbar: display a progress bar.

  Returns:
    A numpy array representing the output video.
  """
  # select number of frames per second
  video_fps = 8.
  # select height of the image
  figure_height = 500
  # number of time steps of the given video
  steps = video.shape[0]
  # estimate duration of the video (in seconds)
  duration = steps / video_fps
  # estimate top_k probabilities and corresponding labels
  top_probs, top_labels, _ = get_top_k_streaming_labels(probs, k=top_k)

  images = []
  step_generator = tqdm.trange(steps) if use_progbar else range(steps)
  for i in step_generator:
    image = plot_streaming_top_preds_at_step(
        top_probs=top_probs,
        top_labels=top_labels,
        step=i,
        image=video[i],
        duration_seconds=duration,
        figure_height=figure_height,
    )
    images.append(image)

  return np.array(images)

init_states = model.init_states(x_values[tf.newaxis].shape)


# predict할 비디오 넣기
video = x_values
images = tf.split(video[tf.newaxis], video.shape[0], axis=1)

all_logits = []

# To run on a video, pass in one frame at a time
states = init_states
for image in tqdm.tqdm(images):
  # predictions for each frame
  logits, states = model({**states, 'image': image})
  all_logits.append(logits)

# concatenating all the logits
logits = tf.concat(all_logits, 0)
# estimating probabilities
probs = tf.nn.softmax(logits, axis=-1)
     

final_probs = probs[-1]
print('Top_k predictions and their probablities\n')
for label, p in get_top_k(final_probs):
  print(f'{label:20s}: {p:.3f}')
  
# Generate a plot and output to a video tensor
plot_video = plot_streaming_top_preds(probs, video, video_fps=8.)