# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:18:08 2020

@author: morenodu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

DS_t_max_test = DS_t_max.where(DS_y['yield'].mean('time') > -0.1 )


y = DS_t_max_test.tmx.sel(time = DS_t_max.indexes['time'].month.isin([8])).to_dataframe().groupby(['time']).mean().values
# y.index = list(range(len(y.index.values)))
X= list(range(len(y)))
X = np.reshape(X, (len(X), 1))

pf = PolynomialFeatures(degree=3)
Xp = pf.fit_transform(X)

md2 = LinearRegression()
md2.fit(Xp, y)
trendp = md2.predict(Xp)

plt.plot(X, y)
plt.plot(X, trendp)
plt.legend(['data', 'polynomial trend'])
plt.show()

detrpoly = [y[i] - trendp[i] for i in range(0, len(y))] + np.mean(y)
plt.plot(X, detrpoly)
plt.title('polynomially detrended data')
plt.show()

plt.plot(df_clim_avg_features.index, y)
plt.plot(df_clim_avg_features.index, df_clim_avg_features['tmx8'].values)
plt.plot(df_clim_avg_features.index, detrpoly)
plt.show()

r2 = r2_score(y, y)
rmse = np.sqrt(mean_squared_error(y, y))
print('r2:', r2)
print('rmse', rmse)

r2 = r2_score(y, detrpoly)
rmse = np.sqrt(mean_squared_error(y, detrpoly))
print('r2:', r2)
print('rmse', rmse)

r2 = r2_score(y, df_clim_avg_features['tmx8'].values)
rmse = np.sqrt(mean_squared_error(y, df_clim_avg_features['tmx8'].values))
print('r2:', r2)
print('rmse', rmse)