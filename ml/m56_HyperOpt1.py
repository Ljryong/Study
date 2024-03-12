import numpy as np
import hyperopt as hp
import pandas as pd
print(hp.__version__)           # 0.2.7

from hyperopt import hp , fmin , tpe , Trials , STATUS_OK

search_space = { 'x1' : hp.quniform('x1',-10,10,1),          # hp.quniform(label,low,high,q) uniform = 균등분포
                'x2' : hp.quniform('x2',-15,15,1),
                }

# hp.quniform(label,low,high,q) : label로 지정된 입력 값 변수 검색 공간을
#                              최소값 low에서 high까지 q의 간격을 가지고 설정
# hp.uniform(label,low,high) : 최소값 low 에서 최대값 high까지 정규분호의 형태의 검색 공간 설정
# hp.quniform(label,upper) : 0부터 최대값 upper 까지 random한 정수 값으로 검색 공간 설정
# hp.lohuniform(label,low,high) : exp(uniform(low,high))값을 반환하며, 반환값의 log변환 된 값은 정규분포 형태를 가지는 검색 공간 설정
# hp.lohuniform(label,low,high) : y 값을 변환할 때 많이 사용한다 , high가 너무 높을 때 log로 값을 줄일 때 좋음(x에서도 사용가능)

def objective_func(search_space) : 
    x1 = search_space['x1']
    x2 = search_space['x2']
    return_value = x1**2 -20*x2
    
    return return_value

trial_val = Trials()

max_eval = 20

best = fmin(                # fit 이라고 생각하면 됨
    fn = objective_func,
    space = search_space,
    algo=tpe.suggest,       # 알고리즘, 디폴트 (그냥 이거 넣는다고 생각해도 무방)
    max_evals = max_eval,           # 서치 횟수
    trials=trial_val,
    rstate = np.random.default_rng(seed=10)      # rstate = random_state
    # rstate = 333      # 이렇게 넣으면 error 함수 자체에서 받아들이지 못함
    
    
)

# print(best)         # {'x': 0.0, 'y': 15.0} = 최적의 파라미터 

# print(trial_val.results)        # 20개 = 시도 결과 
# [{'loss': -216.0, 'status': 'ok'}, {'loss': -175.0, 'status': 'ok'}, {'loss': 129.0, 'status': 'ok'}, {'loss': 200.0, 'status': 'ok'}, 
#  {'loss': 240.0, 'status': 'ok'}, {'loss': -55.0, 'status': 'ok'}, {'loss': 209.0, 'status': 'ok'}, {'loss': -176.0, 'status': 'ok'}, 
#  {'loss': -11.0, 'status': 'ok'}, {'loss': -51.0, 'status': 'ok'}, {'loss': 136.0, 'status': 'ok'}, {'loss': -51.0, 'status': 'ok'}, 
#  {'loss': 164.0, 'status': 'ok'}, {'loss': 321.0, 'status': 'ok'}, {'loss': 49.0, 'status': 'ok'}, {'loss': -300.0, 'status': 'ok'}, 
#  {'loss': 160.0, 'status': 'ok'}, {'loss': -124.0, 'status': 'ok'}, {'loss': -11.0, 'status': 'ok'}, {'loss': 0.0, 'status': 'ok'}]

# print(trial_val.vals)        # 시도에서 선택된 변수 값
# {'x': [-2.0, -5.0, 7.0, 10.0, 10.0, 5.0, 7.0, -2.0, -7.0, 7.0, 4.0, -7.0, -8.0, 9.0, -7.0, 0.0, -0.0, 4.0, 3.0, -0.0],
#  'y': [11.0, 10.0, -4.0, -5.0, -7.0, 4.0, -8.0, 9.0, 3.0, 5.0, -6.0, 5.0, -5.0, -12.0, 0.0, 15.0, -8.0, 7.0, 1.0, 0.0]}

# 최적의 파라미터 출력
print("Best parameters:")
best_params = pd.DataFrame(best, index=['Value'])
print(best_params)

# 시도 결과 출력
print("\nTrial results:")
trial_results = pd.DataFrame(trial_val.results)
print(trial_results)

# 시도에서 선택된 변수 값 출력
print("\nSelected variable values for each trial:")
selected_values = pd.DataFrame(trial_val.vals)

# 각 시도별로 선택된 변수 값 출력
target = [aaa['loss'] for aaa in trial_val.results]
print(target)

df = pd.DataFrame({'target' : target ,
                   'x1' : trial_val.vals['x1'],
                   'x2' : trial_val.vals['x2'],
                   })

print(df)