param_bounds = {'x1' : (-1,5),
                'x2' : (0 , 4)}


def y_function(x1,x2) :
    return - x1 ** 2 - (x2 - 2) **2 + 10

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = y_function,                             # 이 함수의 최댓값을 찾겠다
    pbounds=param_bounds,                       # 파라미터의 범위
    random_state=777,
    
)

optimizer.maximize(init_points= 5,              # 5번
                   n_iter=20,                   # 20번 훈련
                   )

print(optimizer.max)                            # 가장 좋은 결과치만 뽑아 내는게 .max 이다
# {'target': 9.999987989731194, 'params': {'x1': -0.0009484747349116024, 'x2': 1.9966667336614297}}