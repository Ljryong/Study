from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

path = "C:\_data\kaggle\\fat\\"
train_csv = pd.read_csv(path+"train.csv", index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)

# print(train_csv.shape, test_csv.shape, submit_csv.shape)    # (20758, 17) (13840, 16) (13840, 2)
# for label in train_csv:
#         print(train_csv[label].isna().sum())    # 결측치 없음을 확인

class_label = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS','NObeyesdad']


test_csv.loc[test_csv['CALC'] == 'Always', 'CALC'] = 'Frequently'
# print(train_csv.head)

# print(train_csv.columns)
# ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
#        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
#        'CALC', 'MTRANS', 'NObeyesdad']
x_labelEncoder = LabelEncoder()
train_csv['Gender'] = x_labelEncoder.fit_transform(train_csv['Gender'])
train_csv['family_history_with_overweight'] = x_labelEncoder.fit_transform(train_csv['family_history_with_overweight'])
train_csv['FAVC'] = x_labelEncoder.fit_transform(train_csv['FAVC'])
train_csv['CAEC'] = x_labelEncoder.fit_transform(train_csv['CAEC'])
train_csv['SMOKE'] = x_labelEncoder.fit_transform(train_csv['SMOKE'])
train_csv['SCC'] = x_labelEncoder.fit_transform(train_csv['SCC'])
train_csv['CALC'] = x_labelEncoder.fit_transform(train_csv['CALC'])
train_csv['MTRANS'] = x_labelEncoder.fit_transform(train_csv['MTRANS'])

x_labelEncoder = LabelEncoder()
test_csv['Gender'] = x_labelEncoder.fit_transform(test_csv['Gender'])
test_csv['family_history_with_overweight'] = x_labelEncoder.fit_transform(test_csv['family_history_with_overweight'])
test_csv['FAVC'] = x_labelEncoder.fit_transform(test_csv['FAVC'])
test_csv['CAEC'] = x_labelEncoder.fit_transform(test_csv['CAEC'])
test_csv['SMOKE'] = x_labelEncoder.fit_transform(test_csv['SMOKE'])
test_csv['SCC'] = x_labelEncoder.fit_transform(test_csv['SCC'])
test_csv['CALC'] = x_labelEncoder.fit_transform(test_csv['CALC'])
test_csv['MTRANS'] = x_labelEncoder.fit_transform(test_csv['MTRANS'])

y_labelEncoder = LabelEncoder()
train_csv['NObeyesdad'] = y_labelEncoder.fit_transform(train_csv['NObeyesdad'])
# print(train_csv.head)

''' BMI 컬럼 추가 '''
train_csv['BMI'] = train_csv['Weight'] / (train_csv['Height']*train_csv['Height'])
test_csv['BMI'] = test_csv['Weight'] / (test_csv['Height']*test_csv['Height'])

''' 이상치 제거 '''
age_q1 = train_csv['Age'].quantile(0.25)
age_q3 = train_csv['Age'].quantile(0.75)
age_gap = (age_q3 - age_q1 ) * 1.5
age_under = age_q1 - age_gap
age_upper = age_q3 + age_gap
train_csv = train_csv[train_csv['Age']>=age_under]
train_csv = train_csv[train_csv['Age']<=age_upper]

weight_q1 = train_csv['Weight'].quantile(0.25)
weight_q3 = train_csv['Weight'].quantile(0.75)
weight_gap = (weight_q3 - weight_q1 ) * 1.5
weight_under = weight_q1 - weight_gap
weight_upper = weight_q3 + weight_gap
train_csv = train_csv[train_csv['Weight']>=weight_under]
train_csv = train_csv[train_csv['Weight']<=weight_upper]

x = train_csv.drop(['NObeyesdad'], axis=1) # P검정에 의거하여 FAVC와 SMOKE 제거
y = train_csv['NObeyesdad']


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

y = LabelEncoder().fit_transform(y)
y = y.reshape(-1,1)
y = OneHotEncoder(sparse=False).fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    stratify=y
)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (15747, 17) (3937, 17) (15747, 7) (3937, 7)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,x_train.shape[1]],name='x')
y = tf.compat.v1.placeholder(tf.float32, shape=[None,y_train.shape[1]],name='y')

w = tf.compat.v1.Variable(tf.random_normal([x_train.shape[1],y_train.shape[1]]),name='weight')
b = tf.compat.v1.Variable(tf.zeros([1,y_train.shape[1]]),name='bias')

hypothesis = tf.nn.softmax(tf.add(tf.matmul(x,w),b))

loss_fn = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis-y),axis=1)) # categorical_cross_entropy
loss_fn = tf.compat.v1.losses.softmax_cross_entropy(y,hypothesis)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

train = optimizer.minimize(loss_fn)

EPOCHS = 10000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1,EPOCHS+1):
        _, loss = sess.run([train,loss_fn],feed_dict={x:x_train,y:y_train})
        if step%100 == 0:
            print(f"{step}epo loss:{loss}")
        
    pred = sess.run(hypothesis,feed_dict={x:x_test,y:y_test})
    argmax_pred = np.argmax(pred,axis=1)
    # print(argmax_pred.shape)
argmax_y_test = np.argmax(y_test,axis=1)
# print(argmax_y_test.shape)    

from sklearn.metrics import accuracy_score
acc = accuracy_score(argmax_pred,argmax_y_test)
print("ACC: ",acc)
