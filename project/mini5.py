import json
import os
import shutil
import math
import pandas as pd
import cv2
import glob
import numpy as np

path = 'D:/minipro/'

video_start_end_info = []
json_start_end_info = []
error_video = []

def normalization(array):
    scale_array = array.copy()
    scale_array[::2] /= 2048 # x좌표
    scale_array[1::2] /= 1152 # y좌표
    return scale_array

def preprocessing():
    for file in os.listdir(path):
        path_tmp = path + '/' + file + '/'

        kp_path = path_tmp + 'json(키포인트)/'
        new_kp_path = path_tmp + 'txt(키포인트)_norm/'
        
        json_files = glob.glob(os.path.join(kp_path,'**','*.json'), recursive=True)
        # 정렬 필요
        json_files.sort()
        
        os.makedirs(new_kp_path, exist_ok=True) # 폴더 만들고
        for json_file in zip(json_files):
            txt_file_name = os.path.basename(os.path.dirname(json_file))
            f = open(new_kp_path+txt_file_name+'.txt', 'a')
            # 저장
            data = "%s, " %os.path.basename(json_file)[:-5]          # keypoints 값에 normalization 적용해서
            f.write(data)
            f.close()


def edit_json():
    #json_df = pd.DataFrame(json_start_end_info)

    for json_info in json_start_end_info:
        json_dir = json_info[0][:-4]
        json_start = json_info[1]
        json_end = json_info[2]
        for i, json_file_name in enumerate(sorted(os.listdir(json_dir))):
            if i <= json_start * 30 or json_end * 30 < i: # 종료시간보다 전이거나 후면 삭제!
                os.remove(json_dir + '/' + json_file_name)

def edit_video():
    #video_df = pd.DataFrame(video_start_end_info)
    
    for file1 in os.listdir(path):
        path_tmp1 = path
        
        if os.path.isdir(path_tmp1):
            # 'SEN', 'WORD'
            for file in os.listdir(path_tmp1):
                path_tmp2 = path_tmp1 + '/'
                
                img_dir = path_tmp2 + '/img'
                os.makedirs(img_dir)
                
                for video_info in video_start_end_info:
                    # video_info[0] : 영상 경로, video_info[1] : 영상 시작, video_info[2] : 영상 끝

                    if file1 in video_info[0] and file in video_info[0]:
                        #print(file1, file2, video_info[0])
                        editing_video = video_info[0].split('/')[-1]
                        output_dir = img_dir + '/' + editing_video
                        os.makedirs(output_dir)

                        vidcap = cv2.VideoCapture(video_info[0])
                        count = -1
                        while vidcap.isOpened():
                            count+=1
                            success, image = vidcap.read()
                            if success: 
                                if video_info[1] * 30 <= count and count <= video_info[2] * 30:
                                    cv2.imwrite(os.path.join(output_dir, editing_video+ ' %d.jpg') %(count), image)
                            else:
                                cv2.destroyAllWindows()
                                vidcap.release()
            

def edit_files():
    # '가상데이터', '직접촬영데이터'
    for file in os.listdir(path):
        path_tmp1 = path
        
        if os.path.isdir(path_tmp1):
            # 'WORD', 'SEN'
            for file2 in os.listdir(path_tmp1):
                path_tmp2 = path_tmp1 + '/'
                
                # 'json(형태소)', 'mp4', 'json(키포인트)', 
                key = [ path_tmp2 + '/' + file3 + '/' for file3 in os.listdir(path_tmp2)]
                
                morpheme_dir = key[0]
                video_dir = key[1]
                keypoints_dir = key[2]

                for morpheme_file in os.listdir(morpheme_dir):
                    tmp = []
                    tmp2 = []

                    file_name = morpheme_dir + morpheme_file

                    with open(file_name) as json_file:
                        try:
                            json_data = json.load(json_file)
                            clip_name = json_data["metaData"]['name']
                            
                            clip_start = json_data["data"][0]['start']
                            tmp.append(video_dir + clip_name)
                            tmp2.append(keypoints_dir + clip_name)

                            clip_start = math.trunc(clip_start * 10) / 10
                            tmp.append(clip_start)
                            tmp2.append(clip_start)

                            clip_end = json_data["data"][-1]['end']
                            clip_end = math.ceil(clip_end * 10) / 10
                            tmp.append(clip_end)
                            tmp2.append(clip_end)
                        except:
                            print("2. 에러난 파일명: ", clip_name) 
                            
                    video_start_end_info.append(tmp)
                    json_start_end_info.append(tmp2)

    edit_video()
    edit_json()
    
def delete_files():
    # '가상데이터', '직접촬영데이터'
    for file1 in os.listdir(path):
        path_tmp1 = path
        
        if os.path.isdir(path_tmp1):
            # 'SEN', 'WORD'
            for file in os.listdir(path_tmp1):
                path_tmp2 = path_tmp1 + '/' 
                
                # 'json(키포인트)', 'json(형태소)', 'mp4'
                for file3 in os.listdir(path_tmp2):
                    path_tmp3 = path_tmp2 + '/' + file3
                    
                    for file4 in os.listdir(path_tmp3):
                        path_tmp4 = path_tmp3 + '/' + file4
                        filename = file4.split('_')
                        if filename[0] == 'NIA' and not 'F' in filename[4]:
                            #print(path_tmp4, 'remove')
                            try: 
                                shutil.rmtree(path_tmp4)
                            except:
                                os.remove(path_tmp4)

if __name__ == '__main__':
    # F를 제외한 나머지 데이터 삭제
    delete_files()

    # 불필요한 구간(앞, 뒤) 삭제 - 영상(mp4 -> frame), json
    edit_files()

    # pose, face, hand 다 1차원 리스트로 연결
    # confidence 값 제외
    # pose에서 하반신 0 제외
    # 그 외의 0 값 앞 뒤 평균으로 채우기
    # 0~1 scaling
    preprocessing()