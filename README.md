# Time-Series
Applying machine learning methods to predict time series

Даны 100 рядов с 1000 значениями в каждом.
Дисперсия шума ![](http://latex.codecogs.com/gif.latex?%5Csigma%5E2%3D1), дисперсия угла наклона тренда ![](http://latex.codecogs.com/gif.latex?%5Csigma_%7Ba%7D%5E%7B2%7D), средняя частота смены тренда 1 раз за 200 точек (![](http://latex.codecogs.com/gif.latex?%5Clambda%3D200)).
Ошибка излома оценивается как дисперсия суммы 2 нормальных с.в. (независимых углов наклона с обоих сторон от излома) с дисперсией 0.25, т.е. равна 0.5. Чтобы оценить суммарную ошибку для сложного тренда надо ошибку излома умножить на ожидание к-во изломов, т.е. ![](https://latex.codecogs.com/gif.latex?n/%5Clambda) и усреднить, т.е. разделить на n. В итоге получим формулу ![](https://latex.codecogs.com/gif.latex?2%5Csigma_a%5E2/%5Clambda).

### Bounds:
series type | lower bound | upper bound
--- | --- | ---
linear trend | 0 | ![](https://latex.codecogs.com/gif.latex?%5Csigma_a%5E2%3D0.25)
Brownian motion | ![](http://latex.codecogs.com/gif.latex?%5Csigma%5E2%3D1) | -
linear trend with Brownian motion | ![](http://latex.codecogs.com/gif.latex?%5Csigma%5E2%3D1) | ![](https://latex.codecogs.com/gif.latex?%5Csigma%5E2&plus;%5Csigma_a%5E2%3D1.25)
difficult trend | ![](https://latex.codecogs.com/gif.latex?2%5Csigma_a%5E2/%5Clambda%3D0.0025) | ![](https://latex.codecogs.com/gif.latex?%5Csigma_a%5E2&plus;2%5Csigma_a%5E2/%5Clambda%3D0.2525)
difficult trend with Brownian motion | ![sdf](https://latex.codecogs.com/gif.latex?%5Csigma%5E2&plus;2%5Csigma_a%5E2/%5Clambda%3D1.0025) |  ![](https://latex.codecogs.com/gif.latex?%5Csigma%5E2&plus;%5Csigma_a%5E2&plus;2%5Csigma_a%5E2/%5Clambda%3D1.2525)

### Salnikov Dmitry
Есть 2 способа предсказания временных рядов по типу данных: по изначальным значениям и по их разностям ![](https://latex.codecogs.com/gif.latex?x_%7Bi&plus;1%7D-x_i). В случае линейных трендов предсказание по разностям будет оптимальным, им мы и будем пользоваться.

Рассмотрим простую и регуляризованную Lasso линейные регрессии, простое дерево и градиентный бустинг для предсказания последних 300 точек каждого ряда. Сравним ээфективности методов, а заодно посмотрим, как наличие интерсепта в регрессиях влияет на результат.

##### Simple trend
Все методы имеют нулевую ошибку.
Для предсказания в каждом методе использовалось только последнее значение разности.

##### Brownian motion
Кроссвалидация по окнам 2,4,8,12,16,20.
conf int: ![](https://latex.codecogs.com/gif.latex?2sd%28x%29/%5Csqrt%7Bn%7D), n=100 - ширина доверительного интервала в 1 сторону.
Лучшее предсказание --- по предыдущему значению исходных данных или просто нулями для разностей;

name | train | test | conf int
--- | --- | --- | ---
Best | - | 1.0107 | -
Lin regr | 0.991 | 1.015 | 0.018
Lin regr with intercept | 0.993 | 1.017 | 0.018
Lasso | 0.985 | 1.012 | 0.018
Lasso with intercept | 0.986 | 1.014 | 0.018
Tree | 1.010 | 1.048 | 0.020
XGBoost | 0.991 | 1.024 | 0.019

##### Simple trend + Brownian motion
Кроссвалидация по окнам 2,4,8,12,16,20.
Худшее предсказание --- по предыдущему значению исходных данных;
Лучшее --- по предыдущей разности с вычетом среднего значения по всем предыдущим элементам.

name | train | test | conf int
--- | --- | --- | ---
Best | - | 1.007 | -
Worst | - | 1.221 | -
Lin regr | 1.051 | 1.063 | 0.018
Lin regr with intercept | 1.006 | 1.015 | 0.018
Lasso | 1.044 | 1.056 | 0.018
Lasso with intercept | 1.000 | 1.012 | 0.018
Tree | 1.027 | 1.043 | 0.019
XGBoost | 1.005 | 1.022 | 0.018

##### difficult trend
Предсказываем по предыдущей разности.
Лучшее предсказание равно предыдущей разности.

name | train | test | conf int
--- | --- | --- | ---
Best | - | 0.00226 | -
Lin regr | 0.003 | 0.002 | 0.001
Lin regr | 0.003 | 0.002 | 0.001
Lin regr with intercept | 0.056 | 0.002 | 0.001
Lasso | 0.003 | 0.002 | 0.001
Lasso with intercept | 0.026 | 0.002 | 0.001
Tree | 0.061 | 0.057 | 0.034
XGBoost | 0.056 | 0.056 | 0.034

##### difficult trend with Brownian motion
Окна от 2 до 20 с шагом 1 и от 25 до 115 с шагом 5.
Лучшее предсказание --- по предыдущей разности с вычетом среднего значения, посчитанного по последним n элементам, n подбирается кроссвалидацией.
Худшее --- по предыдущему элементу.

name | train | test | conf int
--- | --- | --- | ---
Best | - | 1.027 | -
Worst | - | 1.258 | -
Lin regr | 1.085 | 1.091 | 0.053
Lin regr with intercept | 1.112 | 1.150 | 0.053
Lasso | 1.062 | 1.090 | 0.020
Lasso with intercept | 1.083 | 1.149 | 0.049
Tree | 1.171 | 1.291 | 0.064
XGBoost | 1.099 | 1.181 | 0.023
