from sklearn.ensemble import RandomForestClassifier

# RandomForestClassifier 객체 생성
rf = RandomForestClassifier()

# 현재 설정된 모든 매개변수 확인
default_params = rf.get_params()

# 결과 출력
print(default_params)

# 3.6.9
# {'bootstrap': True,  'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 
#  'max_leaf_nodes': None, 'min_impurity_decrease': 0.0,  'min_samples_leaf': 1, 
#  'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 'warn',  

# 3.9.18
# {'bootstrap': True,  'criterion': 'gini', 'max_depth': None, 
#  'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 
#  'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 
#  }

# max_features , n_estimators 2개 다름

# n_estimators:
# criterion:
# max_depth:
# min_samples_split:
# min_samples_leaf:
# min_weight_fraction_leaf:
# max_features:
# max_leaf_nodes:
# min_impurity_decrease:
# bootstrap: