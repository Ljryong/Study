from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


#1 데이터 
(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()


















