from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer , load_diabetes , load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler , StandardScaler
import pickle
import joblib

#1 데이터
x, y = load_digits(return_X_y=True)
x_train , x_test , y_train , y_test = train_test_split(x,y, random_state= 777 ,test_size= 0.2 ,
                                                       stratify=y
                                                       )

# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2 모델 / 3. 훈련
# model = XGBClassifier()
# path = 'C:/_data/_save/_pickle_test//'
# model = pickle.load(open(path + 'm39_pickle_save.dat' , 'rb' ))           # read binary

path = 'C:/_data/_save/_joblib_test//'
model = joblib.load(path + 'm40_joblib_save.dat'  )

#4 평가
result = model.score(x_test,y_test)
print('최종 점수4 : ' , result )

y_predict = model.predict(x_test)
from sklearn.metrics import accuracy_score 
print('acc : ' , accuracy_score(y_test,y_predict))



# 최종 점수4 :  0.9611111111111111
# acc :  0.9611111111111111


