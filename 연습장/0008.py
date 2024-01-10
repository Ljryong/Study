# 오늘 과제 리스트, 딕셔너리(중괄호로 사용됨), 튜플!
# [] 안에 2개 이상이 있으면 list 1개도 있을순 있다
# a : {} 은 딕셔너리 

# datasets의 결측치를 알아내는 함수 


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_squared_error 
import time
import pandas as pd


#1 데이터

path = "c:/_data/dacon/ddarung//"

train_csv = pd.read_csv(path + "train.csv" , index_col = 0)
test_csv = pd.read_csv(path + "test.csv" , index_col = 0)
submission_csv = pd.read_csv(path + "submission.csv")

print(train_csv)





