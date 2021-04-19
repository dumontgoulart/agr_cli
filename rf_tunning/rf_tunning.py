# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

"""
Created on Mon Apr 19 19:16:07 2021
Steps:

1 - Among the many variables, select the most important (not shown);

2 - Divide the data into two parts: Train and Test;

3 - Use the train data under a stratified cross validation process (5 splits, 6 random reshuffles) to search the grid for best paramters;

4 - Define the best paratmeters based on the accuracy score;

5 - Evaluate performance at the Test data using following scores: 1) Accuracy, 2) Precision, 3) Recall, 4) F-1 Score.
@author: morenodu
"""

# 1 - Load data #####################################
df_clim = pd.read_csv('rf_tuning/df_clim.csv', index_col='year')  
df_yield_us = pd.read_csv('rf_tuning/df_yield.csv', index_col='time')

# Define dataset -> Classification as failure << STD
df_yield = pd.DataFrame( np.where(df_yield_us < df_yield_us.mean() - df_yield_us.std(),True, False), 
                             index = df_yield_us.index, columns = ['severe_loss'] ).astype(int)

# 2 - Divide data
X_train, X_test, y_train, y_test = train_test_split(df_clim, df_yield, test_size=0.3, random_state=0)

# Define models and parameters
model = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight='balanced_subsample')
n_estimators = [400, 600, 700, 1000]
max_features = [2, 3]
max_depth = [4, 5]

# Define grid search
grid = dict(n_estimators = n_estimators, max_features = max_features, max_depth = max_depth)

# 3 - Define how it will be seaarched, being stratified to preserve the two calsses, splitting in 10 and repeating 3 random times
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=0)

# Create model with grid and CV structure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, error_score=0, verbose = 2)

# 4 - Find hyperparameters
grid_result = grid_search.fit(X_train, y_train.values.ravel())

# Summarize results ################################
print("!! Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param)) 
     
    
# 5 - Test model #################################

model_tuned = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight='balanced_subsample',
                                     max_depth = grid_result.best_params_['max_depth'], 
                                     max_features = grid_result.best_params_['max_features'], 
                                     n_estimators = grid_result.best_params_['n_estimators'])

model_tuned.fit(X_train, y_train.values.ravel())

y_pred = model_tuned.predict(X_test)
    
# Basic scores
score_acc = accuracy_score(y_test, y_pred)
score_pcc = precision_score(y_test, y_pred)
score_rec = recall_score(y_test, y_pred)
score_f1 = f1_score(y_test, y_pred)
   
print("Accuracy for Random Forest on test data: ",score_acc)
print("Accuracy for Random Forest on test data: ",score_pcc)
print("Accuracy for Random Forest on test data: ", score_rec)
print("Accuracy for Random Forest on test data: ", score_f1)


