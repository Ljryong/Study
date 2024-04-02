from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

path = "C:\_data\kaggle\\fat\\"
train_csv = pd.read_csv(path+"train.csv", index_col=0)
test_csv = pd.read_csv(path+"test.csv", index_col=0)

class_label = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS','NObeyesdad']

test_csv.loc[test_csv['CALC'] == 'Always', 'CALC'] = 'Frequently'

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

train_csv['BMI'] = train_csv['Weight'] / (train_csv['Height']*train_csv['Height'])
test_csv['BMI'] = test_csv['Weight'] / (test_csv['Height']*test_csv['Height'])

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

x = train_csv.drop(['NObeyesdad'], axis=1)
y = train_csv['NObeyesdad']

y = LabelEncoder().fit_transform(y)
y = y.reshape(-1,1)
y = OneHotEncoder(sparse=False).fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8, stratify=y
)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 17], name='x')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 7], name='y')

w1 = tf.compat.v1.Variable(tf.random_normal([17, 7]), name='weight1')
b1 = tf.compat.v1.Variable(tf.zeros([7]), name='bias1')
layer1 = tf.nn.softmax(tf.add(tf.matmul(x, w1), b1))

w2 = tf.compat.v1.Variable(tf.random_normal([7, 7]), name='weight2')
b2 = tf.compat.v1.Variable(tf.zeros([7]), name='bias2')
hypothesis = tf.nn.softmax(tf.add(tf.matmul(layer1, w2), b2))

loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=hypothesis))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss_fn)

EPOCHS = 1001
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(1, EPOCHS + 1):
        _, loss = sess.run([train, loss_fn], feed_dict={x: x_train, y: y_train})
        if step % 100 == 0:
            print(f"{step} epoch loss: {loss}")

    pred = sess.run(hypothesis, feed_dict={x: x_test, y: y_test})
    argmax_pred = np.argmax(pred, axis=1)

argmax_y_test = np.argmax(y_test, axis=1)
from sklearn.metrics import accuracy_score
acc = accuracy_score(argmax_pred, argmax_y_test)
print("ACC: ", acc)
