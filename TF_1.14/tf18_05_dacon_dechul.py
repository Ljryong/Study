import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings(action='ignore')

path = "C:\\_data\\dacon\\dechul\\"

train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)
submission_csv = pd.read_csv(path+"sample_submission.csv")

train_csv = train_csv[train_csv['주택소유상태'] != 'ANY'] #ANY은딱 한개 존재하기에 그냥 제거
# test_csv = test_csv[test_csv['대출목적'] != '결혼']
test_csv.loc[test_csv['대출목적'] == '결혼' ,'대출목적'] = '기타' #결혼은 제거하면 개수가 안맞기에 기타로 대체



#대출기간 처리
train_csv['대출기간'] = train_csv['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)
test_csv['대출기간'] = test_csv['대출기간'].replace({' 36 months' : 36 , ' 60 months' : 60 }).astype(int)

#근로기간 처리
train_working_time = train_csv['근로기간']
test_working_time = test_csv['근로기간']

for i in range(len(train_working_time)):
    data = train_working_time.iloc[i]
    if data == 'Unknown':
        train_working_time.iloc[i] = np.NaN
    elif data == '10+ years' or data == '10+years':
        train_working_time.iloc[i] = int(30)
    elif data == '< 1 year' or data == '<1 year':
        train_working_time.iloc[i] = int(0)
    else:
        train_working_time.iloc[i] = int(data.split()[0])
    
train_working_time = train_working_time.fillna(train_working_time.mean())

for i in range(len(test_working_time)):
    data = test_working_time.iloc[i]
    if data == 'Unknown':
        test_working_time.iloc[i] = np.NaN
    elif data == '10+ years' or data == '10+years':
        test_working_time.iloc[i] = int(30)
    elif data == '< 1 year' or data == '<1 year':
        test_working_time.iloc[i] = int(0)
    else:
        test_working_time.iloc[i] = int(data.split()[0])
    
test_working_time = test_working_time.fillna(test_working_time.mean())

train_csv['근로기간'] = train_working_time
test_csv['근로기간'] = test_working_time 

#주택소유상태 처리

trian_have_house = train_csv['주택소유상태']
label_encoder = LabelEncoder()
trian_have_house = label_encoder.fit_transform(trian_have_house)
train_csv['주택소유상태'] = trian_have_house

test_have_house = test_csv['주택소유상태']
label_encoder = LabelEncoder()
test_have_house = label_encoder.fit_transform(test_have_house)
test_csv['주택소유상태'] = test_have_house

#대출목적 처리
trian_loan_purpose = train_csv['대출목적']
label_encoder = LabelEncoder()
trian_loan_purpose = label_encoder.fit_transform(trian_loan_purpose)
train_csv['대출목적'] = trian_loan_purpose

test_loan_purpose = test_csv['대출목적']
label_encoder = LabelEncoder()
test_loan_purpose = label_encoder.fit_transform(test_loan_purpose)
test_csv['대출목적'] = test_loan_purpose

#대출등급 처리
train_loan_grade = train_csv['대출등급']
label_encoder = LabelEncoder()
train_loan_grade = label_encoder.fit_transform(train_loan_grade)
train_csv['대출등급'] = train_loan_grade

# print(train_csv.isna().sum(),test_csv.isna().sum(), sep='\n') #결측치 제거 완료 확인함

# for label in train_csv:                                       #모든 데이터가  또는 실수로 변경됨을 확인함
#     for data in train_csv[label]:
#         if type(data) != type(1) and type(data) != type(1.1):
#             print("not int, not float : ",data)


# for label in test_csv:
#     print(label)
#     print(f"train[{label}]: ",np.unique(train_csv[label],return_counts=True))
#     print(f"test[{label}]",np.unique(test_csv[label],return_counts=True))
x = train_csv.drop(['대출등급'],axis=1)
y = train_csv['대출등급']

print(np.unique(y,return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6]), array([16772, 28817, 27622, 13354,  7354,  1954,   420], dtype=int64))

# y = y.to_frame(['대출등급'])
# y = y.reshape(-1,1)

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
# (77034, 13) (19259, 13) (77034, 7) (19259, 7)

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
