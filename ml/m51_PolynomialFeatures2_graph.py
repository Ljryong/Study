import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

plt.rcParams['font.family'] = 'Malgun Gothic'

#1. 데이터
np.random.seed(777)
x = 2 * np.random.rand(100,1) -1        # 0 부터 1까지 100가지 숫자를 생성한다 2 곱하고 1 빼면 -1부터 1까지의 숫자 생성
y = 3 * x**2 + 2 * x + 1 + np.random.rand(100,1)        # y = 3x^2 + 2x + 1 + 노이즈
# print(x)

pf = PolynomialFeatures(degree = 2, include_bias = False )
x_poly = pf.fit_transform(x)
print(x_poly)

#2 모델
model = RandomForestRegressor()
model2 = RandomForestRegressor()

#3 훈련
model.fit(x,y)
model2.fit(x_poly,y)

# 원래 데이터를 그리자!
plt.scatter(x,y,color = 'blue' ,label = '원데이터' )
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ploynomial Regression Example')

x_plot = np.linspace(-1,1,100).reshape(-1,1)
x_plot_poly = pf.transform(x_plot)
y_plot = model.predict(x_plot)
y_plot2 = model2.predict(x_plot_poly)
plt.plot(x_plot,y_plot,color = 'red' ,label = 'Ploynomial Regression Example')
plt.plot(x_plot,y_plot2,color = 'green' ,label = '기냥')
plt.legend()
plt.grid()
plt.show()
