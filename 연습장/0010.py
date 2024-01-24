from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder



#1
path = 'c:\_data\dacon\dechul\\'
train_csv = pd.read_csv(path + 'train.csv',index_col=0)
test_csv = pd.read_csv(path + 'test.csv',index_col=0)
sumission_csv = pd.read_csv(path + 'sample_submission.csv')

# print(train_csv)

led = LabelEncoder() #
led.fit(train_csv['대출기간'])
train_csv['대출기간'] = led.transform(train_csv['대출기간'])
print(train_csv)


