# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:30:00 2021
STACKING
@author: morenodu
"""


# importing module
from sklearn.linear_model import LinearRegression,  Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR

# define dataset
X, y = df_hybrid_us, df_obs_us_det_clip['usda_yield'].values.flatten().ravel()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from skopt import BayesSearchCV
scaler = StandardScaler()
# transform data
X_scaled = pd.DataFrame(scaler.fit_transform(X),columns = X.columns)  #, index = X.index

# # define the search
# grid_search = GridSearchCV(estimator =  ,  param_grid = param_grid, scoring = 'neg_root_mean_squared_error', cv = 5, n_jobs = -1, verbose = 2) 
# grid_search.fit(X_scaled, y)
# print("Best parameters set found on development set:")
# print(grid_search.best_params_)
# means = grid_search.cv_results_["mean_test_score"]
# stds = grid_search.cv_results_["std_test_score"]
# for mean, std, params in zip(means, stds, grid_search.cv_results_["params"]):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))# perform the search

param_grid = {'C': [10], 'epsilon': [1, 0.1, 0.01]}
grid = GridSearchCV(SVR(),param_grid,refit=True,verbose=2)
grid.fit(X_scaled,y)
print(grid.best_estimator_) #

model = SVR(C=10)
model2 = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, 
                                      max_depth = 20, max_features = 'auto',
                                           min_samples_leaf = 1, min_samples_split=2)
# fitting the training data
model.fit(X_scaled,y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=0)

def MBE(y_true, y_pred):
    '''
    Parameters:
        y_true (array): Array of observed values
        y_pred (array): Array of prediction values

    Returns:
        mbe (float): Biais score
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.reshape(len(y_true),1)
    y_pred = y_pred.reshape(len(y_pred),1)   
    diff = (y_true-y_pred)
    mbe = diff.mean()
    return mbe

# Test performance
y_pred = model.predict(X_test.values)

# report performance
print("R2 on test set:", round(r2_score(y_test, y_pred),2))
print("Var score on test set:", round(explained_variance_score(y_test, y_pred),2))
print("MAE on test set:", round(mean_absolute_error(y_test, y_pred),5))
print("RMSE on test set:",round(mean_squared_error(y_test, y_pred, squared=False),5))
print("MBE on test set:", round(MBE(y_test, y_pred),5))
print("______")

#%% Stacking #1 https://www.analyticsvidhya.com/blog/2020/10/how-to-use-stacking-to-choose-the-best-possible-algorithm/

# feature scaling
from sklearn.preprocessing import MinMaxScaler
#to buid models
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
# models for Stacking
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
# to evaluate the model
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error,r2_score
import math
#to find training time of the model
import time
# to visualise al the columns in the dataframe
from lightgbm import LGBMRegressor

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=0)


SVM_model= SVR()
lightgbm = LGBMRegressor()
RF_model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, max_depth = 20, max_features = 'auto', min_samples_leaf = 1, min_samples_split=2)

# Get these models in a list
estimators = [
    # ('RF', RF_model),
               ('MLR', LinearRegression()),
               # ('SVM', SVM_model),
              ] #('LightGBM', lightgbm)
#Stack these models with StackingRegressor
stacking_regressor = StackingRegressor(estimators=estimators)

def plot_regression_results(ax, y_true, y_pred, title, scores, elapsed_time):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('True Yield')
    ax.set_ylabel('Predicted Yield')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    title = title + ' in {:.2f} sec'.format(elapsed_time)
    ax.set_title(title)
    
    
fig, axs = plt.subplots(2, 2, figsize=(9, 7))
axs = np.ravel(axs)
errors_list=[]
for ax, (name, est) in zip(axs, estimators + [('Stacking Regressor',
                                               stacking_regressor)]):
    start_time = time.time()
    model = est.fit(X_train, y_train)
                     
    elapsed_time = time.time() - start_time
    
    pred = model.predict(X_test)
    errors = y_test - model.predict(X_test)
    errors_list.append(errors)
    test_r2= r2_score(np.exp(y_test), np.exp(pred))
    print(f"{name} R2 test score is {test_r2} in {elapsed_time} seconds")
    
    test_rmsle=math.sqrt(mean_squared_log_error(y_test,pred))
    plot_regression_results(ax,y_test,pred,name,(r'$R^2={:.3f}$' + '\n' + 
                            r'$RMSLE={:.3f}$').format(test_r2,test_rmsle),elapsed_time)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()


# titles = ['Random_Forest','Lasso','Gradient_boosting','Stacked_regressor'] 
# f,a = plt.subplots(2,2)
# a = a.ravel()
# for idx,ax in enumerate(a):
#     ax.hist(errors_list[idx])
#     ax.set_title(titles[idx])
# plt.tight_layout()


from sklearn.inspection import PartialDependenceDisplay

features_to_plot = [0,6]
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
disp1 = PartialDependenceDisplay.from_estimator(stacking_regressor, X_scaled, features_to_plot, pd_line_kw={'color':'k'},percentiles=(0.01,0.99), ax = ax1)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.show()

#%% Stacking 2 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=0)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=0))
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
rf_model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, max_depth = 20, max_features = 'auto',
                                           min_samples_leaf = 1, min_samples_split=2)


#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   

averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


#%%
def get_models():
	models = dict()
	models['mlr'] = LinearRegression()
	models['rf'] = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, 
                                      max_depth = 20, max_features = 'auto',
                                           min_samples_leaf = 1, min_samples_split=2)
	return models

def evaluate_model(model, X, y):
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
	return scores

# compare machine learning models for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from matplotlib import pyplot
 
# get the models to evaluate
models = get_models()


# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X_scaled, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

#%%
from sklearn.ensemble import StackingRegressor
def get_stacking():
    
	# define the base models
    level0 = list()
    level0.append(('mlr',LinearRegression()))
    level0.append(('rf', RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, 
                                      max_depth = 20, max_features = 'auto',
                                           min_samples_leaf = 1, min_samples_split=2)))
	# define meta learner model
    level1 = RandomForestRegressor()
	# define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model


def get_models():
	models = dict()
	models['mlr'] = LinearRegression()
	models['rf'] = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, 
                                      max_depth = 20, max_features = 'auto',
                                           min_samples_leaf = 1, min_samples_split=2)
	models['stacking'] = get_stacking()
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=1)
	scores = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=1, error_score='raise')
	return scores
 
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X_scaled, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


model.fit(X_scaled, y)
features_to_plot = [0,6]
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
disp1 = PartialDependenceDisplay.from_estimator(model, X_scaled, features_to_plot, pd_line_kw={'color':'r'},percentiles=(0.01,0.99), ax = ax1)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.show()

#%%
from math import sqrt
from numpy import hstack
from numpy import vstack
from numpy import asarray
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
 
# create a list of base-models
def get_models():
	models = list()
	models.append(LinearRegression())
	models.append(RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1, 
                                      max_depth = 20, max_features = 'auto',
                                           min_samples_leaf = 1, min_samples_split=2))
	return models
 
# collect out of fold predictions form k-fold cross validation
def get_out_of_fold_predictions(X, y, models):
	meta_X, meta_y = list(), list()
	# define split of data
	kfold = KFold(n_splits=10, shuffle=True)
	# enumerate splits
	for train_ix, test_ix in kfold.split(X):
		fold_yhats = list()
		# get data
		train_X, test_X = X.iloc[train_ix], X.iloc[test_ix]
		train_y, test_y = y[train_ix], y[test_ix]
		meta_y.extend(test_y)
		# fit and make predictions with each sub-model
		for model in models:
			model.fit(train_X, train_y)
			yhat = model.predict(test_X)
			# store columns
			fold_yhats.append(yhat.reshape(len(yhat),1))
		# store fold yhats as columns
		meta_X.append(hstack(fold_yhats))
	return vstack(meta_X), asarray(meta_y)
 
# fit all base models on the training dataset
def fit_base_models(X, y, models):
	for model in models:
		model.fit(X, y)
 
# fit a meta model
def fit_meta_model(X, y):
	model = LinearRegression()
	model.fit(X, y)
	return model
 
# evaluate a list of models on a dataset
def evaluate_models(X, y, models):
	for model in models:
		yhat = model.predict(X)
		mse = mean_squared_error(y, yhat)
		r2 = round(r2_score(y, yhat),2)
		print('%s: RMSE %.3f' % (model.__class__.__name__, sqrt(mse)))
		print('%s: R2 %.3f' % (model.__class__.__name__, r2))
 
# make predictions with stacked model
def super_learner_predictions(X, models, meta_model):
	meta_X = list()
	for model in models:
		yhat = model.predict(X)
		meta_X.append(yhat.reshape(len(yhat),1))
	meta_X = hstack(meta_X)
	# predict
	return meta_model.predict(meta_X)
 
# create the inputs and outputs

# split
X_scaled_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
print('Train', X_scaled_train.shape, y_train.shape, 'Test', X_val.shape, y_val.shape)
# get models
models = get_models()
# get out of fold predictions
meta_X, meta_y = get_out_of_fold_predictions(X_scaled_train, y_train, models)
print('Meta ', meta_X.shape, meta_y.shape)
# fit base models
fit_base_models(X_scaled_train, y_train, models)
# fit the meta model
meta_model = fit_meta_model(meta_X, meta_y)
# evaluate base models
evaluate_models(X_val, y_val, models)
# evaluate meta model
yhat = super_learner_predictions(X_val, models, meta_model)
print('Super Learner: RMSE %.3f' % (sqrt(mean_squared_error(y_val, yhat))))
print('Super Learner: R2 %.3f' % (sqrt(r2_score(y_val, yhat))))




