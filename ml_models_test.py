# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:24:44 2021

@author: morenodu
"""
import sklearn
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, \
    GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import scipy
from scipy.stats import uniform, loguniform

#!conda install -y -c conda-forge  xgboost 
import xgboost
from xgboost import XGBRegressor
from xgboost import plot_importance

import lightgbm
from lightgbm import LGBMRegressor

from itertools import product
from datetime import datetime, timedelta

# always use same RANDOM_STATE k-folds for comparability between tests, reproducibility
RANDOMSTATE = 42
np.random.seed(RANDOMSTATE)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

kfolds = KFold(n_splits=10, shuffle=True, random_state=RANDOMSTATE)

MEAN_RESPONSE=y_test.mean()
    
lr = LinearRegression()

# evaluate using kfolds
scores = -cross_val_score(lr, X_train, y_train,
                          scoring="neg_root_mean_squared_error",
                          cv=kfolds,
                          n_jobs=-1)
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print("CV RMSE %.04f (STD %.04f)" % (np.mean(scores), np.std(scores)))
print("R2 on test set:",round(r2_score(y_test, y_pred),2))




#%% Tune elasticnet search space for alphas and L1_ratio
# predictor selection used to create the training set used lasso
# so l1 parameter is close to 0
# could use ridge (eg elasticnet with 0 L1 regularization)
# but then only 1 param, more general and useful to do this with elasticnet
print("ElasticnetCV")

# make pipeline
# with regularization must scale predictors
elasticnetcv = make_pipeline(RobustScaler(),
                             ElasticNetCV(max_iter=100000, 
                                          l1_ratio=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
                                          alphas=np.logspace(-4, -2, 9),
                                          cv=kfolds,
                                          n_jobs=-1,
                                          verbose=1,
                                         ))

#train and get hyperparams
elasticnetcv.fit(X_train, y_train)
l1_ratio = elasticnetcv._final_estimator.l1_ratio_
alpha = elasticnetcv._final_estimator.alpha_
print('l1_ratio', l1_ratio)
print('alpha', alpha)

# evaluate using kfolds on full dataset
# I don't see API to get CV error from elasticnetcv, so we use cross_val_score
elasticnet = make_pipeline(RobustScaler(),
                           ElasticNet(alpha=alpha,
                                      l1_ratio=l1_ratio,
                                      max_iter=10000)
                          )

scores = -cross_val_score(elasticnet, X_train, y_train,
                          scoring="neg_root_mean_squared_error",
                          cv=kfolds,
                          n_jobs=-1)

y_pred = elasticnetcv.predict(X_test)

print("CV RMSE %.04f (STD %.04f)" % (np.mean(scores), np.std(scores)))
print("R2 on test set:",round(r2_score(y_test, y_pred),2))

#%%
gs = make_pipeline(RobustScaler(),
                   GridSearchCV(ElasticNet(max_iter=100000),
                                param_grid={'l1_ratio': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
                                            'alpha': np.logspace(-4, -2, 9),
                                           },
                                scoring='neg_root_mean_squared_error',
                                refit=True,
                                cv=kfolds,
                                n_jobs=-1,
                                verbose=1
                               ))

# do cv using kfolds on full dataset
print("\nCV on full dataset")
gs.fit(X_train, y_train)
print('best params', gs._final_estimator.best_params_)
# print('best score', -gs._final_estimator.best_score_, cv_to_raw(-gs._final_estimator.best_score_))
l1_ratio = gs._final_estimator.best_params_['l1_ratio']
alpha = gs._final_estimator.best_params_['alpha']

scores = -cross_val_score(gs, X_train, y_train,
                          scoring="neg_root_mean_squared_error",
                          cv=kfolds,
                          n_jobs=-1)


y_pred = gs.predict(X_test)

print("Log1p CV RMSE %.06f" % (-gs._final_estimator.best_score_))
print("R2 on test set:",round(r2_score(y_test, y_pred),2))


# eval similarly to before
elasticnet = make_pipeline(RobustScaler(),
                           ElasticNet(alpha=alpha,
                                      l1_ratio=l1_ratio,
                                      max_iter=100000)
                          )
print(elasticnet)
elasticnet.fit(X_train, y_train)

scores = -cross_val_score(elasticnet, X_train, y_train,
                          scoring="neg_root_mean_squared_error",
                          cv=kfolds,
                          n_jobs=-1)


y_pred = elasticnet.predict(X_test)

print("CV RMSE %.04f (STD %.04f)" % (np.mean(scores), np.std(scores)))
print("R2 on test set:",round(r2_score(y_test, y_pred),2))


# small difference in average CV scores reported by GridSearchCV and cross_val_score
# with same alpha, l1_ratio, kfolds
# we used simple average, GridSearchCV is weighted by # of samples per fold?


#%%
# RandomizedSearch
rs = make_pipeline(RobustScaler(),
                   RandomizedSearchCV(ElasticNet(max_iter=100000),
                                      {'alpha': loguniform(0.0001, 0.1),
                                       'l1_ratio': uniform(0.001, 0.5),
                                      },
                                      random_state=RANDOMSTATE,
                                      scoring='neg_root_mean_squared_error',
                                      refit=True,
                                      cv=kfolds,
                                      n_iter=200,
                                      n_jobs=-1,
                                      verbose=1,
                                     )
                  )

# do cv using kfolds on full dataset
print("\nCV on full dataset")
rs.fit(X_train, y_train)
print('best params', rs._final_estimator.best_params_)
print('best score', -rs._final_estimator.best_score_)
l1_ratio = rs._final_estimator.best_params_['l1_ratio']
alpha = rs._final_estimator.best_params_['alpha']

y_pred = rs.predict(X_test)

print("Log1p CV RMSE %.06f" % (-rs._final_estimator.best_score_))
print("R2 on test set:",round(r2_score(y_test, y_pred),2))

# eval similarly to before
elasticnet = make_pipeline(RobustScaler(),
                           ElasticNet(alpha=alpha,
                                      l1_ratio=l1_ratio,
                                      max_iter=100000)
                          )
print(elasticnet)

scores = -cross_val_score(elasticnet, X_train, y_train,
                          scoring="neg_root_mean_squared_error",
                          cv=kfolds,
                          n_jobs=-1)

print()
print("Log1p CV RMSE %.06f (STD %.04f)" % (np.mean(scores), np.std(scores)))





#%%
# evaluate the model
model = XGBRegressor(objective='reg:squarederror')
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
# fit the model on the whole dataset
model = XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 on test set:",round(r2_score(y_test, y_pred),2))

# evaluate the model
model = LGBMRegressor()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model = LGBMRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R2 on test set:",round(r2_score(y_test, y_pred),2))


# evaluate the model
from catboost import CatBoostRegressor
model = CatBoostRegressor()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# fit the model on the whole dataset
model = CatBoostRegressor(verbose=0, n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 on test set:",round(r2_score(y_test, y_pred),2))

# Further catboost


model = CatBoostRegressor(verbose=0)

grid = {'iterations': [100, 150, 200],
        'learning_rate': [0.03, 0.1, 0.11, 0.12],
        'depth': [2, 4, 6, 8],
        'l2_leaf_reg': [1, 3]}

grid_search_result = model.grid_search(grid, 
                                       X=X_train, 
                                       y=y_train, 
                                       plot=False)

pred = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, pred)))
r2 = r2_score(y_test, pred)
print("Testing performance")
print('RMSE: {:.2f}'.format(rmse))
print('R2: {:.2f}'.format(r2))




#%%

model = LGBMRegressor()
# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R2 on test set:",round(r2_score(y_test, y_pred),2))


	
# explore lightgbm tree depth effect on performance
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import LGBMClassifier
from matplotlib import pyplot
# get a list of models to evaluate
def get_models():
	models = dict()
	trees = [10, 50, 100, 500, 1000, 5000]
	for n in trees:
		models[str(n)] = LGBMRegressor(n_estimators=n)
	return models
 
# evaluate a give model using cross-validation
def evaluate_model(model):
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1,error_score='raise')
	return scores
 

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


def get_models():
	models = dict()
	for i in range(1,11):
		models[str(i)] = LGBMRegressor(max_depth=i, num_leaves=2**i)
	return models


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

# get a list of models to evaluate
def get_models():
	models = dict()
	rates = [0.0001, 0.001, 0.01, 0.1, 1.0]
	for r in rates:
		key = '%.4f' % r
		models[key] = LGBMRegressor(learning_rate=r)
	return models

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


# get a list of models to evaluate
def get_models():
	models = dict()
	types = ['gbdt', 'dart', 'goss']
	for t in types:
		models[t] = LGBMRegressor(boosting_type=t)
	return models


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()




t = 'gbdt'
r = 0.0001
i = 10
n =1000


model = LGBMRegressor(boosting_type =t , learning_rate = r, max_depth=i, num_leaves=2**i, n_estimators=n )
# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R2 on test set:",round(r2_score(y_test, y_pred),2))


	







