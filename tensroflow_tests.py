# -*- coding: utf-8 -*-
"""
Checking tensorflow

Created on Fri Dec  3 17:59:38 2021

@author: morenodu
"""
import os
os.chdir('C:/Users/morenodu/OneDrive - Stichting Deltares/Documents/PhD/paper_hybrid_agri/data')
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from  scipy import signal 
import seaborn as sns
import pickle

import tensorflow as tf
physical_device = tf.config.experimental.list_physical_devices('GPU')
print(f'Device found : {physical_device}') 
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

#%% Test sample from TF    
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset = loadtxt('C:/Users/morenodu/Downloads/pima-indians-diabetes.data.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

#%% Apply into case study
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

X = pd.read_csv('dataset_input_hybrid_forML.csv', index_col=[0,1,2])
y = pd.read_csv('dataset_obs_yield_forML.csv', index_col=[0,1,2])

from sklearn.model_selection import train_test_split

# X, y = df_hybrid_us.copy(), df_obs_us_det_clip['usda_yield'].copy().values.flatten().ravel()

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Separate the test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# Split the remaining data to train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, shuffle=True)

#%%
# define the keras model
model = Sequential()
model.add(Dense(800, input_dim=7,kernel_regularizer=regularizers.l2(0.0001), activation='relu')) #,kernel_regularizer=regularizers.l2(0.0001)
model.add(Dropout(0.2))
model.add(Dense(800, kernel_regularizer=regularizers.l2(0.0001),activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(800, kernel_regularizer=regularizers.l2(0.0001),activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
# compile the keras model
model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mean_squared_error','mean_absolute_error'])
# fit the keras model on the dataset
history  = model.fit(x=X_train, y=y_train, epochs=100, batch_size=100, verbose=1, validation_data=(X_val, y_val))
# evaluate the keras model
_, metric_mse, metric_mae = model.evaluate(X_test, y_test)
print('Mean absolute error: %.2f' % (metric_mae))
print('Root Mean squared error: %.2f' % (np.sqrt(metric_mse)))


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  # plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)

plot_loss(history)

def plot_scatter(x, y):
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()
  
  #%% 2nd trial using Batch normalization
from keras.layers import BatchNormalization

# define the keras model
model = Sequential()
model.add(Dense(800, input_dim=7, activation='relu')) #,kernel_regularizer=regularizers.l2(0.0001)
model.add(BatchNormalization())
# model.add(Dense(800, activation='relu'))
# model.add(BatchNormalization())
model.add(Dense(800, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))
# compile the keras model
model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mean_squared_error','mean_absolute_error'])
# fit the keras model on the dataset
history  = model.fit(x=X_train, y=y_train, epochs=200, batch_size=100, verbose=2, validation_data=(X_val, y_val))
# evaluate the keras model
_, metric_mse, metric_mae = model.evaluate(X_test, y_test)
print('Mean absolute error: %.2f' % (metric_mae))
print('Root Mean squared error: %.2f' % (np.sqrt(metric_mse)))


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

plot_loss(history)


#%% COMBINING DIFFERENT MODELS
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import pathlib
import shutil
import tempfile

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"

BATCH_SIZE = 50
STEPS_PER_EPOCH = len(X_train)//BATCH_SIZE


lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

def compile_and_fit(model, name, optimizer=None, max_epochs=2000):
  if optimizer is None:
    optimizer = get_optimizer()
  model.compile(optimizer=optimizer,
                loss='mean_absolute_error',
                metrics=['mean_squared_error','mean_absolute_error'])

  model.summary()

  history = model.fit( x = X_train, y = y_train,
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs = max_epochs,
    validation_data=(X_val, y_val),
    callbacks=get_callbacks(name),
    verbose=0)
  return history

#%% Test new setup

tiny_model = tf.keras.Sequential([
    Dense(16, activation='relu', input_dim = 7),
    Dense(1)
])

size_histories = {}

size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')


plotter = tfdocs.plots.HistoryPlotter(metric = 'mean_absolute_error', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.2, 0.7])

# #Small model
# small_model = tf.keras.Sequential([
#     # `input_shape` is only required here so that `.summary` works.
#     Dense(16, activation='elu', input_dim = 7),
#     Dense(16, activation='elu'),
#     Dense(1)
# ])

# size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')

# Medium model
medium_model = tf.keras.Sequential([
    Dense(64, activation='elu', input_dim = 7),
    Dense(64, activation='elu'),
    Dense(64, activation='elu'),
    Dense(1)
])

size_histories['Medium']  = compile_and_fit(medium_model, "sizes/Medium")

# first amin model
main_model = Sequential()
main_model.add(Dense(800, input_dim=7, activation='relu')) #,kernel_regularizer=regularizers.l2(0.0001)
main_model.add(BatchNormalization())
# main_model.add(Dense(800, activation='relu'))
# main_model.add(BatchNormalization())
main_model.add(Dense(800, activation='relu'))
main_model.add(BatchNormalization())
main_model.add(Dense(1, activation='linear'))


size_histories['main']  = compile_and_fit(medium_model, "sizes/main")

# Second main model: with regularizarion and dropouts
main_model_2 = Sequential()
main_model_2.add(Dense(800, input_dim=7,kernel_regularizer=regularizers.l2(0.0001), activation='relu')) #,kernel_regularizer=regularizers.l2(0.0001)
main_model_2.add(Dropout(0.5))
main_model_2.add(Dense(800, kernel_regularizer=regularizers.l2(0.0001),activation='relu'))
main_model_2.add(Dropout(0.5))
main_model_2.add(Dense(1, activation='linear'))

size_histories['main_2']  = compile_and_fit(medium_model, "sizes/main_2")

plotter.plot(size_histories)
# a = plt.xscale('log')
# plt.xlim([5, max(plt.xlim())])
# plt.hline(0.263)
plt.ylim([0.2, 0.7])
plt.xlabel("Epochs")

size_histories_2 = {}


#%% Hyperparameters runnning
import keras_tuner as kt

def model_builder(hp):
  model = Sequential()
  model.add(Dense(512, input_dim=7, activation='relu')) #,kernel_regularizer=regularizers.l2(0.0001)
  model.add(BatchNormalization())
  # model.add(Dense(800, activation='relu'))
  # model.add(BatchNormalization())
  model.add(Dense(512, activation='relu'))
  model.add(BatchNormalization())
  if hp.Boolean("dropout"):
        model.add(Dropout(rate=0.25))
  model.add(Dense(1, activation='linear'))
  
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])
  
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='mean_absolute_error',
                metrics=['mean_squared_error','mean_absolute_error'])

  return model

tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                     max_epochs=400,
                     factor=3,
                     executions_per_trial=2,
                     overwrite=True,
                     directory="my_dir",
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)

tuner.search(X_train, y_train, epochs=400, validation_data=(X_val, y_val), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

#%% Create structure without pipeline
from scikeras.wrappers import KerasRegressor
from sklearn.inspection import PartialDependenceDisplay

X_projection = pd.read_csv('dataset_future_input_hybrid_forML.csv', index_col=[0,1,2])
X_projection_scale = scaler.transform(X_projection)

def create_model():
   model = Sequential()
   model.add(Dense(800, input_dim=7, activation='relu')) #,kernel_regularizer=regularizers.l2(0.0001)
   model.add(BatchNormalization())
   # model.add(Dense(800, activation='relu'))
   # model.add(BatchNormalization())
   model.add(Dense(800, activation='relu'))
   model.add(BatchNormalization())
   model.add(Dense(1, activation='linear'))
   # compile the keras model
   model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mean_squared_error','mean_absolute_error'])
   return model

model = KerasRegressor(model=create_model, epochs=200, batch_size=64, verbose=1)
# evaluate using 10-fold cross validation
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks = [stop_early])

plot_loss(history)

y_pred = model.predict(X_test)
# report performance
print("R2 on test set:", round(r2_score(y_test, y_pred),2))
print("Var score on test set:", round(explained_variance_score(y_test, y_pred),2))
print("MAE on test set:", round(mean_absolute_error(y_test, y_pred),5))
print("RMSE on test set:",round(mean_squared_error(y_test, y_pred, squared=False),5))
print("______")


predict_ukesm_rcp85 = model.predict(X_projection_scale.values)

features_to_plot = [0,6]
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
disp1 = PartialDependenceDisplay.from_estimator(model, X_projection_scale, features_to_plot, pd_line_kw={'color':'r'},percentiles=(0.01,0.99),  ax = ax1)
disp2 = PartialDependenceDisplay.from_estimator(model, X_train, features_to_plot, ax = disp1.axes_,percentiles=(0.01,0.99),  pd_line_kw={'color':'k'})
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
disp1.plot(ax=[ax1, ax2], line_kw={"label": "Extrapolation", "color": "red"})
disp2.plot(ax=[ax1, ax2], line_kw={"label": "Training", "color": "black"})
ax1.set_ylim(1.0, 2.8)
ax2.set_ylim(1.0, 2.8)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
ax1.legend()
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
disp1.plot(ax=[ax1, ax2], line_kw={"label": "Extrapolation", "color": "red"})
disp2.plot(ax=[ax1, ax2], line_kw={"label": "Training", "color": "black"})
ax1.set_ylim(0, 2.8)
ax2.set_ylim(0, 2.8)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
ax1.legend()
plt.show()
#%% Create pipeline connecting keras with scikit-learn

from scikeras.wrappers import KerasRegressor
from sklearn.inspection import PartialDependenceDisplay

import os
os.environ['PYTHONHASHSEED']= '123'
os.environ['TF_CUDNN_DETERMINISTIC']= '1'

import numpy as np
import tensorflow as tf
import random as python_random

np.random.seed(1)
python_random.seed(1)
tf.random.set_seed(1)

X_projection = pd.read_csv('dataset_future_input_hybrid_forML.csv', index_col=[0,1,2])
# X_projection_scale = scaler.transform(X_projection)

def create_model():
   model = Sequential()
   model.add(Dense(500, input_dim=7, activation='relu')) #,kernel_regularizer=regularizers.l2(0.0001)
   model.add(BatchNormalization())
   # model.add(Dense(800, activation='relu'))
   # model.add(BatchNormalization())
   model.add(Dense(500, activation='relu'))
   model.add(BatchNormalization())
   model.add(Dense(1, activation='linear'))
   # compile the keras model
   model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['mean_squared_error','mean_absolute_error'])
   return model

# stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('estimator', KerasRegressor(model=create_model, epochs=200, batch_size=128, verbose=1))
])

model.fit(X_train, y_train)

# plot_loss(history)

y_pred = model.predict(X_test)
# report performance
print("R2 on test set:", round(r2_score(y_test, y_pred),2))
print("Var score on test set:", round(explained_variance_score(y_test, y_pred),2))
print("MAE on test set:", round(mean_absolute_error(y_test, y_pred),5))
print("RMSE on test set:",round(mean_squared_error(y_test, y_pred, squared=False),5))
print("______")


predict_ukesm_rcp85 = model.predict(X_projection)

features_to_plot = [0,6]
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), dpi=500)
disp1 = PartialDependenceDisplay.from_estimator(model, X_projection, features_to_plot, pd_line_kw={'color':'r'},percentiles=(0.01,0.99),  ax = ax1)
disp2 = PartialDependenceDisplay.from_estimator(model, X_train, features_to_plot, ax = disp1.axes_,percentiles=(0.01,0.99),  pd_line_kw={'color':'k'})
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
disp1.plot(ax=[ax1, ax2], line_kw={"label": "Extrapolation", "color": "red"})
disp2.plot(ax=[ax1, ax2], line_kw={"label": "Training", "color": "black"})
ax1.set_ylim(1.0, 2.8)
ax2.set_ylim(1.0, 2.8)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
ax1.legend()
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
disp1.plot(ax=[ax1, ax2], line_kw={"label": "Extrapolation", "color": "red"})
disp2.plot(ax=[ax1, ax2], line_kw={"label": "Training", "color": "black"})
ax1.set_ylim(0, 2.8)
ax2.set_ylim(0, 2.8)
plt.setp(disp1.deciles_vlines_, visible=False)
plt.setp(disp2.deciles_vlines_, visible=False)
ax1.legend()
plt.show()

#%%
from sklearn.model_selection import cross_validate

def mlp_model(X, Y):

    estimator=MLPRegressor()
    
    
    param_grid = {'hidden_layer_sizes': [(100,100,100), (200,200,200), (800)], #(100,100,100), (200,200,200), (800)
              'activation': ['relu'],
              'alpha': [0.05],
              'learning_rate': ['constant'],
              'solver': ['adam']}
    
    gsc = GridSearchCV(
        estimator,
        param_grid,
        cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    
    grid_result = gsc.fit(X, Y)
    
    
    best_params = grid_result.best_params_
    
    best_mlp = MLPRegressor(hidden_layer_sizes = best_params["hidden_layer_sizes"], 
                            activation =best_params["activation"],
                            solver=best_params["solver"],
                            max_iter= 5000, n_iter_no_change = 200
                  )
    
    scoring = {
               'abs_error': 'neg_mean_absolute_error',
               'squared_error': 'neg_mean_squared_error',
               'r2':'r2'}
    
    scores = cross_validate(best_mlp, X, Y, cv=5, scoring=scoring, return_train_score=True, return_estimator = True)
    print(scores)
    return scores

scores_test = mlp_model(X,y.values.ravel())

scores_test['estimator'].to_csv('results_setup_test_mlp.csv')
scores_test['test_abs_error'].to_csv('results_scores_test_mlp.csv')

# # fit model
# history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0, batch_size=32)
# # evaluate the model
# _, train_acc = model.evaluate(trainX, trainy, verbose=0)
# _, test_acc = model.evaluate(testX, testy, verbose=0)
# print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# # plot training history
# pyplot.plot(history.history['accuracy'], label='train')
# pyplot.plot(history.history['val_accuracy'], label='test')
# pyplot.legend()
# pyplot.show()