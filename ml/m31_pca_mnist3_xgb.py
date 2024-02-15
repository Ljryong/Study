# xgboost로 그리드서치 , 랜덤서치 , Halving중 1개 사용

from keras.datasets import mnist
from sklearn.decomposition import PCA
import time
from sklearn.model_selection import train_test_split , GridSearchCV , RandomizedSearchCV
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
warnings.warn('ignore')

(x_train, y_train ) , (x_test, y_test ) = mnist.load_data()

# tree_method = 'gpu_hist'
# predictor = 'gpu_predictor'
# gpu_id = 0

parameters =[
    {'n_estimators' : [100,200] ,'max_depth':[6,10,12],'min_samples_leaf' : [3,10]},
    {'max_depth': [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3,5,7,10],'min_samples_split' : [2,3,5,10]},
    {'min_samples_split' : [2,3,5,10] },
    {'min_samples_split' : [2,3,5,10]}
]

x_train = x_train.reshape(x_train.shape[0] , x_train.shape[1]*x_train.shape[2] )
x_test = x_test.reshape(x_test.shape[0] , x_test.shape[1]*x_test.shape[2] )

# print(x_train.shape)
# print(x_test.shape)


number = [154,331,486,713,784]
for i ,number in enumerate(number) :
    pca = PCA(n_components=number)
    x_train2 = pca.fit_transform(x_train)
    x_test2 = pca.transform(x_test)
    # evr = pca.explained_variance_ratio_
    
    model = GridSearchCV(xgb.XGBClassifier(tree_method = 'hist', device='cuda') ,
                               parameters,
                            #    cv = 3 ,
                               refit = True , 
                               n_jobs = -1 )
    

    start = time.time()
    model.fit(x_train2,y_train)
    end = time.time()

    result = model.score(x_test2,y_test)

    print( '결과',i+1 ,'PCA=', number )
    print('걸린 시간',end-start)
    print('acc',result )


# 결과 1 PCA= 154
# 걸린 시간 30.1601459980011
# acc 0.9391000270843506

# 결과 2 PCA= 331
# 걸린 시간 31.5673885345459
# acc 0.9144999980926514

# 결과 3 PCA= 486
# 걸린 시간 33.23084259033203
# acc 0.9117000102996826

# 결과 4 PCA= 713
# 걸린 시간 35.7853729724884
# acc 0.911899983882904

# 결과 5 PCA= 784
# 걸린 시간 36.44583511352539
# acc 0.9103000164031982










